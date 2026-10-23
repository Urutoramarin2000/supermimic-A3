"""Batched differentiable forward kinematics (positions, rotations, velocities) built from an MjModel.

The world model predicts only the generalized state (root pose/twist, joint angles/velocities), as in the
SuperMimic web demo where the network state is MuJoCo's qpos/qvel. Rewards, terminations and observations
are analytic functions of that state, so their gradients flow through this module.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from . import rotations as rot


class Kinematics:
  def __init__(self, mjm: mujoco.MjModel, body_names: list[str], site_names: list[str], device):
    nb = mjm.nbody
    depth = np.zeros(nb, int)
    for b in range(1, nb):
      depth[b] = depth[mjm.body_parentid[b]] + 1
    assert mjm.jnt_type[mjm.body_jntadr[1]] == mujoco.mjtJoint.mjJNT_FREE, "body 1 must be the floating root"
    # Bodies are processed level by level (all bodies of one depth in one batched op).
    order = [b for d in range(2, depth.max() + 1) for b in range(nb) if depth[b] == d]
    new_index = {1: 0}
    for b in order:
      new_index[b] = len(new_index)
    self.levels = []
    for d in range(2, depth.max() + 1):
      bodies = [b for b in order if depth[b] == d]
      pos, lrot, axis, qadr = [], [], [], []
      for b in bodies:
        pos.append(mjm.body_pos[b])
        lrot.append(rot.quat_to_mat(torch.tensor(mjm.body_quat[b], dtype=torch.float64)).numpy())
        nj = mjm.body_jntnum[b]
        assert nj <= 1, f"body {mjm.body(b).name} has {nj} joints"
        if nj == 1:
          j = mjm.body_jntadr[b]
          assert mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
          assert np.allclose(mjm.jnt_pos[j], 0), "joint anchor must be the body origin"
          axis.append(mjm.jnt_axis[j])
          qadr.append(mjm.jnt_qposadr[j] - 7)
        else:
          axis.append(np.zeros(3))
          qadr.append(0)
      t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt, device=device)
      k = rot.skew(t(axis))
      self.levels.append({
        "parent": t([new_index[mjm.body_parentid[b]] for b in bodies], torch.long),
        "pos": t(pos), "lrot": t(lrot), "axis": t(axis), "k": k, "k2": rot.mm3(k, k),
        "qadr": t(qadr, torch.long),
      })
    self.body_ids = torch.tensor([new_index[mjm.body(n).id] for n in body_names], device=device)
    self.site_body = torch.tensor([new_index[mjm.site_bodyid[mjm.site(n).id]] for n in site_names], device=device)
    self.site_pos = torch.tensor(np.array([mjm.site_pos[mjm.site(n).id] for n in site_names]),
                                 dtype=torch.float32, device=device)
    self.site_rot = rot.quat_to_mat(torch.tensor(np.array([mjm.site_quat[mjm.site(n).id] for n in site_names]),
                                                 dtype=torch.float32, device=device))
    self.num_out = len(body_names) + len(site_names)

  def __call__(self, root_pos, root_rot, q, root_vel=None, root_angvel_body=None, qd=None):
    """root_pos [N,3], root_rot [N,3,3], q [N,nq_joints]; optional world root velocity, body-frame root
    angular velocity and joint velocities. Returns (pos [N,K,3], rot [N,K,3,3], vel, angvel) for the
    requested bodies followed by the requested sites (velocities in world frame, None if not asked)."""
    with_vel = qd is not None
    p_all, r_all = root_pos[:, None], root_rot[:, None]
    if with_vel:
      v_all = root_vel[:, None]
      w_all = rot.mv3(root_rot, root_angvel_body)[:, None]
    for lv in self.levels:
      pr, pp = r_all[:, lv["parent"]], p_all[:, lv["parent"]]
      angle = q[:, lv["qadr"]][..., None, None]
      joint = torch.eye(3, device=q.device) + torch.sin(angle) * lv["k"] + (1 - torch.cos(angle)) * lv["k2"]
      r = rot.mm3(pr, rot.mm3(lv["lrot"], joint))
      p = pp + rot.mv3(pr, lv["pos"])
      p_all, r_all = torch.cat([p_all, p], 1), torch.cat([r_all, r], 1)
      if with_vel:
        pw = w_all[:, lv["parent"]]
        w = pw + rot.mv3(r, lv["axis"]) * qd[:, lv["qadr"]][..., None]
        v = v_all[:, lv["parent"]] + torch.cross(pw, p - pp, dim=-1)
        v_all, w_all = torch.cat([v_all, v], 1), torch.cat([w_all, w], 1)
    sr = r_all[:, self.site_body]
    sp = p_all[:, self.site_body] + rot.mv3(sr, self.site_pos)
    pos = torch.cat([p_all[:, self.body_ids], sp], 1)
    rmat = torch.cat([r_all[:, self.body_ids], rot.mm3(sr, self.site_rot)], 1)
    if not with_vel:
      return pos, rmat, None, None
    sw = w_all[:, self.site_body]
    sv = v_all[:, self.site_body] + torch.cross(sw, sp - p_all[:, self.site_body], dim=-1)
    return pos, rmat, torch.cat([v_all[:, self.body_ids], sv], 1), torch.cat([w_all[:, self.body_ids], sw], 1)
