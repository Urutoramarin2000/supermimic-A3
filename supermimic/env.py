"""Simulator-side data collection (Algorithm A.1 steps 2-3): roll out the policy with action noise in
MuJoCo Warp, push transitions to the replay buffer, reset with return-based sampling."""

from __future__ import annotations

import torch

from . import state as st
from .robot import NUM_ACTIONS
from .task import OBS_DIM, XI_DIM


class Collector:
  def __init__(self, cfg, sim, task, lib, sampler, device, randomize: bool = True):
    self.cfg, self.sim, self.task, self.lib, self.sampler = cfg, sim, task, lib, sampler
    self.device, self.randomize = device, randomize
    n = sim.num_envs
    self.n = n
    self.motion = torch.zeros(n, dtype=torch.long, device=device)
    self.frame = torch.zeros(n, dtype=torch.long, device=device)
    self.start_bin = torch.zeros(n, dtype=torch.long, device=device)
    self.episode = torch.zeros(n, dtype=torch.long, device=device)
    self.ep_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.disc_return = torch.zeros(n, device=device)
    self.xi = torch.zeros(n, XI_DIM, device=device)
    self.prev_action = torch.zeros(n, NUM_ACTIONS, device=device)
    self.hist = torch.zeros(n, cfg.hist_len, OBS_DIM, device=device)
    self.stats = {"ep_len": [], "ep_norm_return": [], "terminated": [], "reward": []}
    self.ep_reward = torch.zeros(n, device=device)
    self.reset(torch.arange(n, device=device))

  def state(self) -> torch.Tensor:
    return st.from_qpos_qvel(self.sim.qpos, self.sim.qvel)

  def reset(self, ids: torch.Tensor, motion=None, frame=None):
    c, k = self.cfg, len(ids)
    if motion is None:
      bins, motion, frame = self.sampler.sample(k)
      self.start_bin[ids] = bins
    self.motion[ids], self.frame[ids] = motion, frame
    idx = self.lib.index(motion, frame)
    self.sim.set_state(ids, self.lib.qpos[idx], self.lib.qvel[idx])
    xi = torch.zeros(k, XI_DIM, device=self.device)
    if self.randomize:
      xi[:, 0] = c.friction_lo + (c.friction_hi - c.friction_lo) * torch.rand(k, device=self.device)
      xi[:, 1:1 + NUM_ACTIONS] = c.calib_range * (2 * torch.rand(k, NUM_ACTIONS, device=self.device) - 1)
      span = torch.tensor([c.com_x, c.com_y, c.com_z], device=self.device)
      xi[:, -3:] = span * (2 * torch.rand(k, 3, device=self.device) - 1)
    else:
      xi[:, 0] = 1.0
    self.xi[ids] = xi
    self.sim.randomize(ids, xi[:, 0], xi[:, -3:])
    self.episode[ids] += 1
    self.ep_steps[ids] = 0
    self.disc_return[ids] = 0
    self.ep_reward[ids] = 0
    self.prev_action[ids] = 0
    s = st.from_qpos_qvel(self.lib.qpos[idx], self.lib.qvel[idx])
    o = self.task.obs(s, self.prev_action[ids], xi)
    self.hist[ids] = self.task.add_noise(o)[:, None].expand(-1, c.hist_len, -1)

  @torch.no_grad()
  def step(self, actor, replay=None, noise: float | None = None, finish_cb=None):
    c, task = self.cfg, self.task
    noise = c.action_noise if noise is None else noise
    s = self.state()
    goal = task.goal(s, self.motion, self.frame)
    action = actor(self.hist, goal)
    if noise > 0:
      action = action + noise * torch.randn_like(action)
    if replay is not None:
      replay.add(s, action, self.xi, self.motion, self.frame, self.episode)
    self.sim.step(task.joint_targets(action, self.xi))
    s1 = self.state()
    bad = ~torch.isfinite(s1).all(-1)
    s1 = torch.where(bad[:, None], s, s1)
    reward = task.reward(s1, self.motion, self.frame + 1, action, self.prev_action)
    self.frame += 1
    terminated = task.terminated(s1, self.motion, self.frame) | bad
    truncated = self.lib.ended(self.motion, self.frame)
    self.disc_return += (c.gamma ** self.ep_steps.float()) * reward
    self.ep_reward += reward
    self.ep_steps += 1
    self.prev_action = action
    o = task.add_noise(task.obs(s1, action, self.xi))
    self.hist = torch.cat([self.hist[:, 1:], o[:, None]], 1)
    done = terminated | truncated
    ids = done.nonzero()[:, 0]
    if len(ids):
      self.sampler.update(self.start_bin[ids], self.disc_return[ids])
      self.stats["ep_len"].append(self.ep_steps[ids].float())
      self.stats["ep_norm_return"].append(self.ep_reward[ids] / (task.max_step_reward * self.ep_steps[ids]))
      self.stats["terminated"].append(terminated[ids].float())
      if finish_cb is not None:
        finish_cb(ids, terminated[ids])
      self.reset(ids)
    self.stats["reward"].append(reward.mean()[None] / task.max_step_reward)
    return reward, terminated, truncated

  def pop_stats(self) -> dict:
    out = {}
    for k, v in self.stats.items():
      if v:
        out[k] = torch.cat(v).mean().item()
    self.stats = {k: [] for k in self.stats}
    return out
