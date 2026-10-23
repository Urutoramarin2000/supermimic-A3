"""Reference motion library (ChingMu -> A3 retarget, 50 Hz) with precomputed body kinematics."""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import torch

from . import rotations as rot
from . import state as st
from .robot import NUM_JOINTS


def _root_angvel_body(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
  r = rot.quat_to_mat(torch.tensor(quat_wxyz, dtype=torch.float64))
  rel = r[:-1].transpose(-1, -2) @ r[1:]  # body-frame increments
  w = rot.log_map(rel).numpy() * fps
  out = np.zeros((len(quat_wxyz), 3))
  out[1:-1] = 0.5 * (w[1:] + w[:-1])
  out[0], out[-1] = w[0], w[-1]
  return out


class MotionLibrary:
  def __init__(self, motion_dir: str, pattern: str, kin, device, bin_seconds: float = 1.0):
    files = sorted(glob.glob(os.path.join(motion_dir, pattern)))
    assert files, f"no motions in {motion_dir}/{pattern}"
    qpos_l, qvel_l, self.names, self.lengths, hits, starts = [], [], [], [], [], []
    total = 0
    for f in files:
      d = np.load(f, allow_pickle=True)
      fps = float(d["fps"][0]) if "fps" in d.files else 50.0
      assert abs(fps - 50.0) < 1e-6, f"{f}: expected 50 Hz, got {fps}"
      qpos = d["qpos"].astype(np.float64)
      assert qpos.shape[1] == 7 + NUM_JOINTS, qpos.shape
      qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
      qvel = np.zeros((len(qpos), 6 + NUM_JOINTS))
      qvel[:, :3] = np.gradient(qpos[:, :3], 1 / fps, axis=0)
      qvel[:, 3:6] = _root_angvel_body(qpos[:, 3:7], fps)
      qvel[:, 6:] = np.gradient(qpos[:, 7:], 1 / fps, axis=0)
      qpos_l.append(qpos)
      qvel_l.append(qvel)
      self.names.append(os.path.basename(f).split("_a3_31dof")[0])
      self.lengths.append(len(qpos))
      starts.append(total)
      hits.append(np.asarray(d["hit_frames"], int) if "hit_frames" in d.files else np.zeros(0, int))
      total += len(qpos)
    self.num_motions = len(files)
    self.num_frames = total
    self.device = device
    self.start = torch.tensor(starts, device=device)
    self.length = torch.tensor(self.lengths, device=device)
    qpos = torch.tensor(np.concatenate(qpos_l), dtype=torch.float32, device=device)
    qvel = torch.tensor(np.concatenate(qvel_l), dtype=torch.float32, device=device)
    self.qpos, self.qvel = qpos, qvel
    self.state = st.from_qpos_qvel(qpos, qvel)  # [F, STATE_DIM]
    p, r, q, v, w, qd = st.split(self.state)
    pos, rm, vel, angvel = [], [], [], []
    for i in range(0, total, 8192):  # tracked bodies + racket site, chunked to bound memory
      out = kin(p[i:i + 8192], r[i:i + 8192], q[i:i + 8192], v[i:i + 8192], w[i:i + 8192], qd[i:i + 8192])
      for lst, x in zip((pos, rm, vel, angvel), out):
        lst.append(x)
    self.body_pos, self.body_rot = torch.cat(pos), torch.cat(rm)
    self.body_vel, self.body_angvel = torch.cat(vel), torch.cat(angvel)
    self.hit_frames = hits
    # 1 s bins for return-based reset sampling.
    bin_len = int(round(bin_seconds * 50))
    bm, bs, bl = [], [], []
    for m, n in enumerate(self.lengths):
      for k in range(math.ceil((n - 1) / bin_len)):
        bm.append(m)
        bs.append(k * bin_len)
        bl.append(min(bin_len, n - 1 - k * bin_len))
    self.bin_motion = torch.tensor(bm, device=device)
    self.bin_start = torch.tensor(bs, device=device)
    self.bin_len = torch.tensor(bl, device=device)
    self.num_bins = len(bm)

  @property
  def seconds(self) -> float:
    return self.num_frames / 50.0

  def index(self, motion: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Global frame index, holding the last frame past the end of a clip."""
    return self.start[motion] + torch.minimum(frame, self.length[motion] - 1)

  def ended(self, motion: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    return frame >= self.length[motion] - 1
