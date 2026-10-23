"""World model, policy (goal encoder + shared decoder) and critic networks (Table A.4)."""

from __future__ import annotations

import torch
from torch import nn

from . import state as st


def mlp(inp: int, hidden, out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
  layers, d = [], inp
  for h in hidden:
    layers.append(nn.Linear(d, h))
    if layer_norm:
      layers.append(nn.LayerNorm(h))
    layers.append(act())
    d = h
  layers.append(nn.Linear(d, out))
  return nn.Sequential(*layers)


class AmpMLP(nn.Module):
  """MLP whose matmuls may run in bf16 autocast; inputs/outputs and everything around it stay fp32."""

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.net = mlp(*args, **kwargs)
    self.amp = False

  def forward(self, x):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
      return self.net(x).float()


class Normalizer(nn.Module):
  """Running mean/std, updated explicitly from simulator data and frozen inside gradient passes."""

  def __init__(self, dim: int, eps: float = 1e-2, clip: float = 10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))
    self.eps, self.clip = eps, clip

  @torch.no_grad()
  def update(self, x: torch.Tensor):
    x = x.reshape(-1, x.shape[-1]).float()
    n = x.shape[0]
    mean, var = x.mean(0), x.var(0, unbiased=False)
    tot = self.count + n
    delta = mean - self.mean
    self.mean += delta * n / tot
    self.var = (self.var * self.count + var * n + delta**2 * self.count * n / tot) / tot
    self.count = torch.clamp(tot, max=1e7)  # bounded count keeps the statistics slowly adaptive

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps**2), -self.clip, self.clip)


class WorldModel(nn.Module):
  """Deterministic dynamics f(s, a, xi) -> s'. Inputs are yaw/translation invariant features; the output
  is a normalized state change in the current heading frame, integrated on the SE(3) x R^n state."""

  def __init__(self, act_dim: int, xi_dim: int, hidden):
    super().__init__()
    inp = st.FEAT_DIM + 2 * act_dim + xi_dim  # features, action, PD error, sim context
    self.net = AmpMLP(inp, hidden, st.DELTA_DIM, act=nn.SiLU, layer_norm=True)
    nn.init.zeros_(self.net.net[-1].weight)
    nn.init.zeros_(self.net.net[-1].bias)
    self.in_norm = Normalizer(inp)
    self.register_buffer("d_mean", torch.zeros(st.DELTA_DIM))
    self.register_buffer("d_std", torch.ones(st.DELTA_DIM))

  def inputs(self, s, action, pd_err, xi):
    return torch.cat([st.features(s), action, pd_err, xi], -1)

  def forward(self, s, action, pd_err, xi):
    x = self.in_norm(self.inputs(s, action, pd_err, xi))
    d = self.net(x) * self.d_std + self.d_mean
    return st.apply_deltas(s, d)

  @torch.no_grad()
  def update_stats(self, s0, s1, action, pd_err, xi, momentum: float = 0.01):
    self.in_norm.update(self.inputs(s0, action, pd_err, xi))
    d = st.deltas(s0, s1)
    if self.in_norm.count <= d.shape[0]:
      self.d_mean.copy_(d.mean(0))
      self.d_std.copy_(d.std(0) + 1e-6)
    else:
      self.d_mean.lerp_(d.mean(0), momentum)
      self.d_std.lerp_(d.std(0) + 1e-6, momentum)


class Actor(nn.Module):
  """a = dec(enc(goal), last H observations); deterministic (Appendix B)."""

  def __init__(self, obs_dim: int, hist_len: int, goal_dim: int, act_dim: int, enc_hidden, dec_hidden,
               latent: int):
    super().__init__()
    self.encoder = AmpMLP(goal_dim, enc_hidden, latent)
    self.decoder = AmpMLP(latent + obs_dim * hist_len, dec_hidden, act_dim)
    self.obs_norm = Normalizer(obs_dim)
    self.goal_norm = Normalizer(goal_dim)
    nn.init.uniform_(self.decoder.net[-1].weight, -1e-3, 1e-3)
    nn.init.zeros_(self.decoder.net[-1].bias)

  def latent(self, goal):
    return torch.tanh(self.encoder(self.goal_norm(goal)))

  def forward(self, obs_hist, goal):
    z = self.latent(goal)
    o = self.obs_norm(obs_hist).flatten(1)
    return self.decoder(torch.cat([z, o], -1))


class Critic(nn.Module):
  def __init__(self, goal_dim: int, xi_dim: int, hidden):
    super().__init__()
    inp = st.FEAT_DIM + goal_dim + xi_dim
    self.net = AmpMLP(inp, hidden, 1)
    self.in_norm = Normalizer(inp)

  def inputs(self, s, goal, xi):
    return torch.cat([st.features(s), goal, xi], -1)

  def forward(self, s, goal, xi):
    return self.net(self.in_norm(self.inputs(s, goal, xi)))[..., 0]
