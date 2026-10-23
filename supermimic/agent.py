"""SuperMimic learner: k-step world-model fit (Eq. 1) and short-horizon BPTT actor-critic through the frozen
world model (Eq. 2, 3), Algorithm A.1 steps 4-8."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from . import state as st
from .networks import Actor, Critic, WorldModel
from .robot import NUM_ACTIONS
from .task import OBS_DIM, XI_DIM


def _requires_grad(module, flag: bool):
  for p in module.parameters():
    p.requires_grad_(flag)


def _allreduce_grads(params):
  if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
    return
  grads = [p.grad for p in params if p.grad is not None]
  flat = torch.cat([g.reshape(-1) for g in grads])
  dist.all_reduce(flat)
  flat /= dist.get_world_size()
  i = 0
  for g in grads:
    g.copy_(flat[i:i + g.numel()].view_as(g))
    i += g.numel()


class SuperMimic:
  def __init__(self, cfg, task, device):
    self.cfg, self.task, self.device = cfg, task, device
    self.wm = WorldModel(NUM_ACTIONS, XI_DIM, cfg.wm_hidden).to(device)
    self.actor = Actor(OBS_DIM, cfg.hist_len, task.goal_dim, NUM_ACTIONS, cfg.enc_hidden, cfg.dec_hidden,
                       cfg.latent_dim).to(device)
    self.critic = Critic(task.goal_dim, XI_DIM, cfg.critic_hidden).to(device)
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    self.lr = cfg.lr * math.sqrt(world)
    self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=self.lr, fused=True)
    self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr, betas=(0.9, 0.95), fused=True)
    self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr, fused=True)
    self.err_scale = st.nominal_scale(device)
    for net in (self.wm, self.actor, self.critic):
      for m in net.modules():
        if hasattr(m, "amp"):
          m.amp = cfg.precision == "bf16"
    self.step_fn, self.window_fn = self._imagine_step, self._window_loss
    if cfg.compile:
      self.step_fn = torch.compile(self._imagine_step, dynamic=False)
      self.window_fn = torch.compile(self._window_loss, dynamic=False)

  def set_lr(self, it: int):
    c = self.cfg
    frac = min(it / max(c.lr_decay_iters, 1), 1.0)
    lr = self.lr * (c.lr_final_ratio + (1 - c.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))
    for opt in (self.wm_opt, self.actor_opt, self.critic_opt):
      for g in opt.param_groups:
        g["lr"] = lr
    return lr

  # ---------------------------------------------------------------------------------------------------------
  def _window_loss(self, states, actions, xi):
    s, loss, errs = states[0], 0.0, []
    for j in range(self.cfg.wm_window):
      s = self.wm(s, actions[j], self.task.pd_error(s, actions[j], xi), xi)
      e = st.state_error(s, states[j + 1], self.err_scale).mean(-1).mean()
      loss = loss + e
      errs.append(e.detach())
    return loss / self.cfg.wm_window, errs[0], errs[-1]

  def wm_update(self, replay) -> dict:
    """Fit f_w on k-step autoregressive windows rolled forward on logged actions (Eq. 1)."""
    c, task = self.cfg, self.task
    acc = torch.zeros(3, device=self.device)
    for i in range(c.wm_minibatches):
      states, actions, xi = replay.sample_windows(c.wm_batch, c.wm_window)
      if i == 0:
        self.wm.update_stats(states[0], states[1], actions[0], task.pd_error(states[0], actions[0], xi), xi)
      loss, e1, ek = self.window_fn(states, actions, xi)
      self.wm_opt.zero_grad(set_to_none=True)
      loss.backward()
      _allreduce_grads(list(self.wm.parameters()))
      torch.nn.utils.clip_grad_norm_(self.wm.parameters(), c.grad_clip)
      self.wm_opt.step()
      acc += torch.stack([loss.detach(), e1, ek])
    acc = (acc / c.wm_minibatches).tolist()
    return {"wm_loss": acc[0], "wm_err1": acc[1], "wm_errk": acc[2]}

  # ---------------------------------------------------------------------------------------------------------
  def _imagine_step(self, s, hist, prev_a, xi, motion, frame):
    task = self.task
    g = task.goal(s, motion, frame)
    a = self.actor(hist, g)
    s1 = self.wm(s, a, task.pd_error(s, a, xi), xi)
    r = task.reward(s1, motion, frame + 1, a, prev_a)
    d = task.soft_terminated(s1, motion, frame + 1)
    o1 = task.add_noise(task.obs(s1, a, xi))
    hist1 = torch.cat([hist[:, 1:], o1[:, None]], 1)
    return s1, hist1, a, r, d, g

  def policy_update(self, replay) -> dict:
    """BPTT through H imagined steps of the frozen world model, critic bootstrap (Eq. 2), GAE critic (Eq. 3)."""
    c, task = self.cfg, self.task
    b = replay.sample_starts(c.imag_batch, c.hist_len)
    xi, motion, frame = b["xi"], b["motion"], b["frame"]
    with torch.no_grad():  # observation history of the start state, with fresh sensor noise
      n, h = b["hist_state"].shape[:2]
      hist_xi = xi[:, None].expand(-1, h, -1).reshape(n * h, -1)
      o = task.obs(b["hist_state"].reshape(n * h, -1), b["hist_prev_action"].reshape(n * h, -1), hist_xi)
      hist = task.add_noise(o).reshape(n, h, -1)
    s, prev_a = b["state"], b["prev_action"]
    _requires_grad(self.wm, False)
    _requires_grad(self.critic, False)
    alive = torch.ones(n, device=self.device)
    ret = torch.zeros(n, device=self.device)
    traj_s, traj_g, traj_r, traj_d = [], [], [], []
    for t in range(c.horizon):
      if c.checkpoint_imagination:
        s1, hist, a, r, d, g = checkpoint(self.step_fn, s, hist, prev_a, xi, motion, frame,
                                          use_reentrant=False)
      else:
        s1, hist, a, r, d, g = self.step_fn(s, hist, prev_a, xi, motion, frame)
      ret = ret + (c.gamma**t) * alive * r
      traj_s.append(s.detach())
      traj_g.append(g.detach())
      traj_r.append(r.detach())
      traj_d.append(d.detach())
      alive = alive * (1 - d)
      s, prev_a, frame = s1, a, frame + 1
    g_h = task.goal(s, motion, frame)
    v_h = self.critic(s, g_h, xi)
    ret = ret + (c.gamma**c.horizon) * alive * v_h
    actor_loss = -ret.mean() / c.horizon
    self.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    _requires_grad(self.wm, True)
    _requires_grad(self.critic, True)
    params = list(self.actor.parameters())
    _allreduce_grads(params)
    gnorm = torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
    torch.nn.utils.clip_grad_value_(params, c.actor_elem_clip)
    self.actor_opt.step()

    # Critic targets: lambda-returns over the imagined transitions only, fixed while the critic is fit.
    with torch.no_grad():
      states = torch.stack(traj_s + [s.detach()])  # [H+1, n, S]
      goals = torch.stack(traj_g + [g_h.detach()])
      values = self.critic(states.flatten(0, 1), goals.flatten(0, 1),
                           xi.repeat(c.horizon + 1, 1)).view(c.horizon + 1, n)
    rewards, dones = torch.stack(traj_r), torch.stack(traj_d)
    targets = torch.zeros_like(rewards)
    nxt = values[-1]
    for t in reversed(range(c.horizon)):
      nxt = rewards[t] + c.gamma * (1 - dones[t]) * ((1 - c.lam) * values[t + 1] + c.lam * nxt)
      targets[t] = nxt
    cs, cg = states[:-1].flatten(0, 1), goals[:-1].flatten(0, 1)
    cx, ct = xi.repeat(c.horizon, 1), targets.flatten()
    self.critic.in_norm.update(self.critic.inputs(cs, cg, cx))
    closs = torch.zeros((), device=self.device)
    for _ in range(c.critic_epochs):
      perm = torch.randperm(len(ct), device=self.device)
      for idx in perm.chunk(c.critic_minibatches):
        v = self.critic(cs[idx], cg[idx], cx[idx])
        loss = ((v - ct[idx]) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        _allreduce_grads(list(self.critic.parameters()))
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
        self.critic_opt.step()
        closs += loss.detach()
    vals = torch.stack([-actor_loss.detach(), rewards.mean() / task.max_step_reward, alive.detach().mean(),
                        gnorm, closs / (c.critic_epochs * c.critic_minibatches), values.mean()]).tolist()
    return dict(zip(("actor_obj", "imag_reward", "imag_alive", "actor_gnorm", "critic_loss", "value"), vals))

  def state_dict(self):
    return {k: getattr(self, k).state_dict() for k in
            ("wm", "actor", "critic", "wm_opt", "actor_opt", "critic_opt")}

  def load_state_dict(self, d):
    for k, v in d.items():
      getattr(self, k).load_state_dict(v)
