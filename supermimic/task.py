"""Motion-tracking interface: observation, goal, reward and termination as differentiable functions of the
state, shared by simulator rollouts and world-model imagination (Appendix A)."""

from __future__ import annotations

import torch

from . import rotations as rot
from . import state as st
from .config import Config
from .robot import NUM_ACTIONS, POLICY_JOINT_IDS

OBS_DIM = 2 * NUM_ACTIONS + 3 + 3 + NUM_ACTIONS  # joint pos, joint vel, IMU gyro, gravity, previous action
XI_DIM = 1 + NUM_ACTIONS + 3  # friction, joint calibration offsets, torso CoM offset


class TrackingTask:
  def __init__(self, cfg: Config, kin, lib, info: dict, device):
    self.cfg, self.kin, self.lib = cfg, kin, lib
    self.pid = torch.tensor(POLICY_JOINT_IDS, device=device)
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    self.default_pos = t(info["default_pos"])
    self.action_scale = t(info["action_scale"])[self.pid]
    lo, hi = t(info["joint_range"][:, 0]), t(info["joint_range"][:, 1])
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * cfg.soft_limit_factor
    self.soft_lo, self.soft_hi = mid - half, mid + half
    self.goal_frame_dim = 2 * NUM_ACTIONS + 6 + (3 if cfg.goal_root_pos else 0)
    self.goal_dim = cfg.goal_frames * self.goal_frame_dim
    self.noise_scale = torch.cat([
      t([cfg.noise_joint_pos] * NUM_ACTIONS), t([cfg.noise_joint_vel] * NUM_ACTIONS),
      t([cfg.noise_ang_vel] * 3), t([cfg.noise_gravity] * 3), torch.zeros(NUM_ACTIONS, device=device)])
    self.dt = cfg.sim_dt * cfg.decimation
    self.max_step_reward = self.dt * (cfg.w_body_pos + cfg.w_body_ori + cfg.w_body_linvel + cfg.w_body_angvel
                                      + cfg.w_root_pos + cfg.w_root_ori + cfg.w_racket_pos + cfg.w_racket_ori)
    self.future = torch.arange(1, cfg.goal_frames + 1, device=device)

  # --- actions -------------------------------------------------------------------------------------------
  def joint_targets(self, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """PD setpoints for all 31 joints (head at its default)."""
    tgt = self.default_pos.expand(action.shape[0], -1).clone()
    tgt[:, self.pid] = self.default_pos[self.pid] + xi[:, 1:1 + NUM_ACTIONS] + self.action_scale * action
    return tgt

  def pd_error(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    q = st.split(s)[2][:, self.pid]
    return self.default_pos[self.pid] + xi[:, 1:1 + NUM_ACTIONS] + self.action_scale * action - q

  # --- observation ---------------------------------------------------------------------------------------
  def obs(self, s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Noise-free proprioception o-bar (the calibration offset shifts the measured joint zero)."""
    _, r, q, _, w, qd = st.split(s)
    jp = q[:, self.pid] - self.default_pos[self.pid] - xi[:, 1:1 + NUM_ACTIONS]
    gravity = -r[:, 2, :]  # R^T (0, 0, -1)
    return torch.cat([jp, qd[:, self.pid], w, gravity, prev_action], -1)

  def add_noise(self, o: torch.Tensor) -> torch.Tensor:
    return o + (2 * torch.rand_like(o) - 1) * self.noise_scale

  # --- goal ----------------------------------------------------------------------------------------------
  def goal(self, s: torch.Tensor, motion: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """K future reference frames expressed relative to the current robot root."""
    idx = self.lib.index(motion[:, None], frame[:, None] + self.future)  # [N, K]
    ref = self.lib.state[idx]
    rp, rr, rq, _, _, rqd = st.split(ref)
    p, r, *_ = st.split(s)
    parts = [rq[..., self.pid], rqd[..., self.pid]]
    if self.cfg.goal_root_pos:
      ht = rot.tr(st.heading(r))[:, None]
      parts.append(rot.mv3(ht, rp - p[:, None]))
    parts.append(rot.rot6d(rot.mm3(rot.tr(r)[:, None], rr)))
    return torch.cat(parts, -1).reshape(s.shape[0], -1)

  # --- reward and termination ----------------------------------------------------------------------------
  def reward(self, s1: torch.Tensor, motion: torch.Tensor, frame1: torch.Tensor, action: torch.Tensor,
             prev_action: torch.Tensor, return_terms: bool = False):
    """BeyondMimic tracking reward for landing in s1 at reference frame frame1 (Table A.2, times dt)."""
    c = self.cfg
    idx = self.lib.index(motion, frame1)
    p, r, q, v, w, qd = st.split(s1)
    bp, br, bv, bw = self.kin(p, r, q, v, w, qd)
    rbp, rbr = self.lib.body_pos[idx], self.lib.body_rot[idx]
    rbv, rbw = self.lib.body_vel[idx], self.lib.body_angvel[idx]
    ref_root_p, ref_root_r = rbp[:, 0], rbr[:, 0]  # first tracked body is the root
    delta_yaw = st.heading(rot.mm3(r, rot.tr(ref_root_r)))[:, None]
    anchor = torch.cat([p[:, :2], ref_root_p[:, 2:]], -1)[:, None]
    des_p = anchor + rot.mv3(delta_yaw, rbp - ref_root_p[:, None])
    des_r = rot.mm3(delta_yaw, rbr)
    nb = bp.shape[1] - 1  # last entry is the racket site
    e_pos = ((des_p - bp) ** 2).sum(-1)
    e_ori = rot.angle_sq(des_r, br)
    e_vel = ((rbv - bv) ** 2).sum(-1)
    e_ang = ((rbw - bw) ** 2).sum(-1)
    terms = {
      "body_pos": c.w_body_pos * torch.exp(-e_pos[:, :nb].mean(-1) / c.std_body_pos**2),
      "body_ori": c.w_body_ori * torch.exp(-e_ori[:, :nb].mean(-1) / c.std_body_ori**2),
      "body_linvel": c.w_body_linvel * torch.exp(-e_vel[:, :nb].mean(-1) / c.std_body_linvel**2),
      "body_angvel": c.w_body_angvel * torch.exp(-e_ang[:, :nb].mean(-1) / c.std_body_angvel**2),
      "root_pos": c.w_root_pos * torch.exp(-((ref_root_p - p) ** 2).sum(-1) / c.std_root_pos**2),
      "root_ori": c.w_root_ori * torch.exp(-rot.angle_sq(ref_root_r, r) / c.std_root_ori**2),
      "action_rate": c.w_action_rate * ((action - prev_action) ** 2).sum(-1),
      "joint_limit": c.w_joint_limit * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
    }
    if c.w_racket_pos:
      terms["racket_pos"] = c.w_racket_pos * torch.exp(-e_pos[:, nb] / c.std_racket_pos**2)
    if c.w_racket_ori:
      terms["racket_ori"] = c.w_racket_ori * torch.exp(-e_ori[:, nb] / c.std_racket_ori**2)
    total = self.dt * sum(terms.values())
    if return_terms:
      return total, {k: self.dt * x for k, x in terms.items()}, e_pos[:, :nb].mean(-1).sqrt()
    return total

  def termination_margin(self, s: torch.Tensor, motion: torch.Tensor, frame: torch.Tensor):
    """(height excess, tilt excess); positive means terminated (Appendix A)."""
    ref = self.lib.state[self.lib.index(motion, frame)]
    p, r, *_ = st.split(s)
    rp, rr, *_ = st.split(ref)
    return (rp[:, 2] - p[:, 2]).abs() - self.cfg.term_height, (rr[:, 2, 2] - r[:, 2, 2]).abs() - self.cfg.term_tilt

  def terminated(self, s, motion, frame) -> torch.Tensor:
    h, t = self.termination_margin(s, motion, frame)
    return (h > 0) | (t > 0)

  def soft_terminated(self, s, motion, frame) -> torch.Tensor:
    """Termination probability d-hat in [0, 1], differentiable."""
    h, t = self.termination_margin(s, motion, frame)
    temp = self.cfg.soft_term_temp
    return 1 - (1 - torch.sigmoid(h / temp)) * (1 - torch.sigmoid(t / temp))
