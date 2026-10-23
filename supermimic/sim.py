"""Batched A3 simulator on MuJoCo Warp with per-env domain randomization (data collection only)."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


class WarpSim:
  def __init__(self, mjm: mujoco.MjModel, info: dict, num_envs: int, decimation: int, device: str,
               nconmax: int | None = None, njmax: int | None = None):
    self.mjm, self.num_envs, self.decimation = mjm, num_envs, decimation
    self.device = torch.device(device)
    wp.init()
    self.wp_device = wp.get_device(device)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    with wp.ScopedDevice(self.wp_device):
      self.m = mjw.put_model(mjm)
      self.d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=nconmax, njmax=njmax)
      # Per-world copies of the randomized model fields (as mjlab's expand_model_fields).
      for name in ("geom_friction", "body_ipos"):
        src = getattr(self.m, name)
        host = src.numpy()[:1]  # [1, n, (vector dims)]
        dst = wp.array(np.repeat(host, num_envs, axis=0), dtype=src.dtype)
        setattr(self.m, name, dst)
      self.reset_mask_wp = wp.zeros(num_envs, dtype=bool)
    tt = wp.to_torch
    self.qpos, self.qvel, self.ctrl = tt(self.d.qpos), tt(self.d.qvel), tt(self.d.ctrl)
    self.geom_friction, self.body_ipos = tt(self.m.geom_friction), tt(self.m.body_ipos)
    self.reset_mask = tt(self.reset_mask_wp)
    self.default_friction = self.geom_friction[0].clone()
    self.default_ipos = self.body_ipos[0].clone()
    self.robot_geoms = torch.tensor(info["robot_geoms"], device=self.device)
    self.torso_body = int(info["torso_body"])
    self.graph = None
    if self.wp_device.is_cuda and wp.is_mempool_enabled(self.wp_device):
      with wp.ScopedDevice(self.wp_device), wp.ScopedCapture() as cap:
        self._substeps()
      self.graph = cap.graph

  def _substeps(self):
    for _ in range(self.decimation):
      mjw.step(self.m, self.d)

  def step(self, joint_targets: torch.Tensor):
    self.ctrl.copy_(joint_targets)
    torch.cuda.synchronize(self.device)
    with wp.ScopedDevice(self.wp_device):
      if self.graph is not None:
        wp.capture_launch(self.graph)
      else:
        self._substeps()
    wp.synchronize_device(self.wp_device)

  def set_state(self, env_ids: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor):
    """Resets solver warm starts etc. of env_ids, then writes their generalized state."""
    self.reset_mask.zero_()
    self.reset_mask[env_ids] = True
    torch.cuda.synchronize(self.device)
    with wp.ScopedDevice(self.wp_device):
      mjw.reset_data(self.m, self.d, reset=self.reset_mask_wp)
    wp.synchronize_device(self.wp_device)
    self.qpos[env_ids] = qpos
    self.qvel[env_ids] = qvel

  def randomize(self, env_ids: torch.Tensor, friction: torch.Tensor, com_offset: torch.Tensor):
    g = self.robot_geoms
    fr = self.default_friction.expand(len(env_ids), -1, -1).clone()
    fr[:, g, 0] = friction[:, None]
    self.geom_friction[env_ids] = fr
    ipos = self.default_ipos.expand(len(env_ids), -1, -1).clone()
    ipos[:, self.torso_body] += com_offset
    self.body_ipos[env_ids] = ipos
