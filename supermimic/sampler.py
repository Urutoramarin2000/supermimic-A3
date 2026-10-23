"""Return-based reset sampling over 1 s motion bins (Appendix D.1)."""

from __future__ import annotations

import torch


class ResetSampler:
  def __init__(self, lib, cfg, max_step_reward: float):
    self.lib, self.cfg = lib, cfg
    n = lib.num_bins
    dev = lib.device
    steps_left = (lib.length[lib.bin_motion] - 1 - lib.bin_start).float()
    self.g_max = max_step_reward * (1 - cfg.gamma**steps_left) / (1 - cfg.gamma)
    self.g = torch.zeros(n, device=dev)
    self.visits = torch.zeros(n, device=dev)
    self.past_w = torch.full((n,), 1.0 / n, device=dev)

  def weights(self) -> torch.Tensor:
    c = self.cfg
    hard = torch.softmax(-c.reset_temp * self.g / self.g_max.clamp(min=1e-6), 0)
    self.past_w.lerp_(hard, 0.01)
    uniform = torch.full_like(hard, 1.0 / len(hard))
    return c.reset_uniform * uniform + c.reset_hard * hard + c.reset_past * self.past_w / self.past_w.sum()

  def sample(self, n: int):
    b = torch.multinomial(self.weights(), n, replacement=True)
    frame = self.lib.bin_start[b] + (torch.rand(n, device=b.device) * self.lib.bin_len[b]).long()
    return b, self.lib.bin_motion[b], frame

  def update(self, bins: torch.Tensor, returns: torch.Tensor):
    """Running estimate of the discounted return from each bin, from finished simulator episodes."""
    if len(bins) == 0:
      return
    s = torch.zeros_like(self.g).index_add_(0, bins, returns)
    k = torch.zeros_like(self.g).index_add_(0, bins, torch.ones_like(returns))
    seen = k > 0
    first = seen & (self.visits == 0)
    mean = s[seen] / k[seen]
    upd = self.g.clone()
    upd[seen] = torch.lerp(self.g[seen], mean, self.cfg.bin_ema)
    upd[first] = (s[first] / k[first])
    self.g = upd
    self.visits += k

  def state_dict(self):
    return {"g": self.g, "visits": self.visits, "past_w": self.past_w}

  def load_state_dict(self, d):
    self.g, self.visits, self.past_w = d["g"], d["visits"], d["past_w"]
