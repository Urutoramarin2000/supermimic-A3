"""Fixed-size GPU replay buffer of simulator transitions; the newest step overwrites the oldest."""

from __future__ import annotations

import torch

from . import state as st


class Replay:
  def __init__(self, length: int, num_envs: int, act_dim: int, xi_dim: int, device):
    self.L, self.N, self.device = length, num_envs, device
    z = lambda *shape, dt=torch.float32: torch.zeros(length, num_envs, *shape, dtype=dt, device=device)
    self.state = z(st.STATE_DIM)
    self.action = z(act_dim)  # executed (noisy) action
    self.xi = z(xi_dim, dt=torch.float16)
    self.motion = z(dt=torch.int32)
    self.frame = z(dt=torch.int32)
    self.episode = z(dt=torch.int64)  # per-env episode counter
    self.t = 0  # total steps written per env

  @property
  def size(self) -> int:
    return min(self.t, self.L)

  def add(self, s, action, xi, motion, frame, episode):
    i = self.t % self.L
    self.state[i], self.action[i], self.xi[i] = s, action, xi.half()
    self.motion[i], self.frame[i], self.episode[i] = motion.int(), frame.int(), episode
    self.t += 1

  def _sample_slots(self, n: int, span: int):
    """Random (time, env) with time in the stored window and room for `span` later steps."""
    lo = self.t - self.size
    hi = self.t - span  # exclusive
    assert hi > lo, "replay too small"
    tt = torch.randint(lo, hi, (n,), device=self.device)
    env = torch.randint(0, self.N, (n,), device=self.device)
    return tt, env

  def sample_windows(self, n: int, k: int):
    """k-step windows inside one episode: states [k+1,n,S], actions [k,n,A], xi [n,X]."""
    out_t, out_e, got = [], [], 0
    for _ in range(20):
      tt, env = self._sample_slots(2 * n, k + 1)
      ep0 = self.episode[tt % self.L, env]
      ep1 = self.episode[(tt + k) % self.L, env]
      ok = ep0 == ep1
      out_t.append(tt[ok])
      out_e.append(env[ok])
      got += int(ok.sum())
      if got >= n:
        break
    tt, env = torch.cat(out_t)[:n], torch.cat(out_e)[:n]
    steps = (tt[None] + torch.arange(k + 1, device=self.device)[:, None]) % self.L
    states = self.state[steps, env[None]]
    actions = self.action[steps[:-1], env[None]]
    return states, actions, self.xi[tt % self.L, env].float()

  def sample_starts(self, n: int, hist: int):
    """Start points for imagination with the last `hist` states/actions of the same episode
    (clamped to the episode start)."""
    tt, env = self._sample_slots(n, 1)
    lo = self.t - self.size
    ep = self.episode[tt % self.L, env]
    offs = torch.arange(-hist + 1, 1, device=self.device)  # oldest .. current
    ht = (tt[:, None] + offs).clamp(min=lo)
    same = self.episode[ht % self.L, env[:, None]] == ep[:, None]
    # clamp to the first step of the episode inside the window
    first = torch.where(same, ht, torch.full_like(ht, 1 << 60)).min(1).values
    ht = torch.maximum(ht, first[:, None])
    hs = self.state[ht % self.L, env[:, None]]  # [n, hist, S]
    # previous action of each history step (zero at the episode start)
    pt = ht - 1
    prev_ok = (pt >= lo) & (self.episode[pt.clamp(min=0) % self.L, env[:, None]] == ep[:, None])
    pa = self.action[pt.clamp(min=0) % self.L, env[:, None]] * prev_ok[..., None]
    i = tt % self.L
    return {
      "hist_state": hs, "hist_prev_action": pa, "state": hs[:, -1], "prev_action": pa[:, -1],
      "xi": self.xi[i, env].float(), "motion": self.motion[i, env].long(), "frame": self.frame[i, env].long(),
    }
