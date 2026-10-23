"""Generalized robot state used by the simulator, the replay buffer and the world model.

A state is one tensor [..., STATE_DIM]:
  root position (3, world) | root rotation (9, world, row-major) | joint angles (31)
  | root linear velocity (3, world) | root angular velocity (3, root frame, MuJoCo convention) | joint velocities (31)
It carries the same information as MuJoCo's qpos (38) / qvel (37).
"""

from __future__ import annotations

import torch

from . import rotations as rot
from .robot import NUM_JOINTS

NJ = NUM_JOINTS
STATE_DIM = 3 + 9 + NJ + 3 + 3 + NJ
FEAT_DIM = 1 + 6 + 3 + 3 + NJ + NJ  # yaw/xy invariant network features
DELTA_DIM = 3 + 3 + NJ + 3 + 3 + NJ

_P, _R, _Q = slice(0, 3), slice(3, 12), slice(12, 12 + NJ)
_V, _W, _QD = slice(12 + NJ, 15 + NJ), slice(15 + NJ, 18 + NJ), slice(18 + NJ, 18 + 2 * NJ)


def split(s: torch.Tensor):
  return (s[..., _P], s[..., _R].reshape(s.shape[:-1] + (3, 3)), s[..., _Q], s[..., _V], s[..., _W], s[..., _QD])


def join(p, r, q, v, w, qd) -> torch.Tensor:
  return torch.cat([p, r.reshape(r.shape[:-2] + (9,)), q, v, w, qd], -1)


def from_qpos_qvel(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
  return join(qpos[..., :3], rot.quat_to_mat(qpos[..., 3:7]), qpos[..., 7:], qvel[..., :3], qvel[..., 3:6], qvel[..., 6:])


def to_qpos_qvel(s: torch.Tensor):
  p, r, q, v, w, qd = split(s)
  return torch.cat([p, rot.mat_to_quat(r), q], -1), torch.cat([v, w, qd], -1)


def heading(r: torch.Tensor) -> torch.Tensor:
  """Yaw-only rotation of the root."""
  return rot.yaw_mat(rot.yaw_of(r))


def features(s: torch.Tensor) -> torch.Tensor:
  p, r, q, v, w, qd = split(s)
  ht = rot.tr(heading(r))
  return torch.cat([p[..., 2:3], rot.rot6d(rot.mm3(ht, r)), rot.mv3(ht, v), w, q, qd], -1)


def deltas(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
  """One-step change expressed in the heading frame of s0 (target of the dynamics network)."""
  p0, r0, q0, v0, w0, qd0 = split(s0)
  p1, r1, q1, v1, w1, qd1 = split(s1)
  ht = rot.tr(heading(r0))
  dp = rot.mv3(ht, p1 - p0)
  dr = rot.mv3(ht, rot.log_map(rot.mm3(r1, rot.tr(r0))))
  dv = rot.mv3(ht, v1 - v0)
  return torch.cat([dp, dr, q1 - q0, dv, w1 - w0, qd1 - qd0], -1)


def apply_deltas(s0: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
  p0, r0, q0, v0, w0, qd0 = split(s0)
  h = heading(r0)
  dp, dr, dq = d[..., 0:3], d[..., 3:6], d[..., 6:6 + NJ]
  dv, dw, dqd = d[..., 6 + NJ:9 + NJ], d[..., 9 + NJ:12 + NJ], d[..., 12 + NJ:]
  p1 = p0 + rot.mv3(h, dp)
  r1 = rot.gram_schmidt(rot.mm3(rot.exp_map(rot.mv3(h, dr)), r0))
  v1 = v0 + rot.mv3(h, dv)
  return join(p1, r1, q0 + dq, v1, w0 + dw, qd0 + dqd)


def state_error(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Per-sample normalized squared state error [..., DELTA_DIM] (root rotation as a rotation vector)."""
  pa, ra, qa, va, wa, qda = split(a)
  pb, rb, qb, vb, wb, qdb = split(b)
  e = torch.cat([pa - pb, rot.log_map(rot.mm3(ra, rot.tr(rb))), qa - qb, va - vb, wa - wb, qda - qdb], -1)
  return (e / scale) ** 2


def nominal_scale(device, pos=0.01, ori=0.02, joint=0.02, lin_vel=0.1, ang_vel=0.2, joint_vel=0.5):
  vals = [pos] * 3 + [ori] * 3 + [joint] * NJ + [lin_vel] * 3 + [ang_vel] * 3 + [joint_vel] * NJ
  return torch.tensor(vals, device=device)
