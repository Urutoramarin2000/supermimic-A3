"""Sanity checks: torch FK vs MuJoCo, state delta round trip, and the simulator holding reference motions.

  python -m scripts.check_model [--device cuda:0] [--xml ... --motion_dir ...]
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np
import torch

from supermimic import state as st
from supermimic.config import get_config
from supermimic.robot import RACKET_SITE, TRACKED_BODIES
from supermimic.task import XI_DIM
from supermimic.train import build


def main():
  cfg = get_config(sys.argv[1:] + ["--num_envs", "256"])
  dev = torch.device(cfg.device)
  mjm, info, kin, lib, task, sim = build(cfg, dev)
  print(f"model: nq={mjm.nq} nv={mjm.nv} nu={mjm.nu} ngeom={mjm.ngeom} robot geoms={len(info['robot_geoms'])}")
  print(f"motions={lib.num_motions} frames={lib.num_frames} ({lib.seconds / 60:.1f} min) bins={lib.num_bins}")

  # 1. FK against MuJoCo on reference frames with random joint velocities.
  rng = np.random.default_rng(0)
  idx = torch.tensor(rng.choice(lib.num_frames, 64, replace=False), device=dev)
  qpos = lib.qpos[idx].double().cpu().numpy()
  qvel = lib.qvel[idx].double().cpu().numpy() + rng.normal(0, 1.0, (64, mjm.nv))
  s = st.from_qpos_qvel(torch.tensor(qpos, device=dev).float(), torch.tensor(qvel, device=dev).float())
  pos, rmat, vel, angvel = kin(*st.split(s))
  d = mujoco.MjData(mjm)
  ids = [mjm.body(n).id for n in TRACKED_BODIES]
  site = mjm.site(RACKET_SITE).id
  err = {"pos": 0.0, "rot": 0.0, "vel": 0.0, "angvel": 0.0}
  res = np.zeros(6)
  for i in range(64):
    d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
    mujoco.mj_forward(mjm, d)
    ref_p = np.concatenate([d.xpos[ids], d.site_xpos[site][None]])
    ref_r = np.concatenate([d.xmat[ids].reshape(-1, 3, 3), d.site_xmat[site].reshape(1, 3, 3)])
    ref_v, ref_w = [], []
    for b in ids:
      mujoco.mj_objectVelocity(mjm, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
      ref_w.append(res[:3].copy())
      ref_v.append(res[3:].copy())
    mujoco.mj_objectVelocity(mjm, d, mujoco.mjtObj.mjOBJ_SITE, site, res, 0)
    ref_w.append(res[:3].copy())
    ref_v.append(res[3:].copy())
    err["pos"] = max(err["pos"], np.abs(pos[i].cpu().numpy() - ref_p).max())
    err["rot"] = max(err["rot"], np.abs(rmat[i].cpu().numpy() - ref_r).max())
    err["vel"] = max(err["vel"], np.abs(vel[i].cpu().numpy() - np.array(ref_v)).max())
    err["angvel"] = max(err["angvel"], np.abs(angvel[i].cpu().numpy() - np.array(ref_w)).max())
  print("FK max abs error vs MuJoCo:", {k: f"{v:.2e}" for k, v in err.items()})
  ok = err["pos"] < 1e-4 and err["rot"] < 1e-4 and err["vel"] < 1e-3 and err["angvel"] < 1e-3

  # 2. State deltas round trip.
  s1 = lib.state[idx + 1]
  back = st.apply_deltas(lib.state[idx], st.deltas(lib.state[idx], s1))
  rt = st.state_error(back, s1, torch.ones(st.DELTA_DIM, device=dev)).max().sqrt().item()
  print(f"delta round-trip max error: {rt:.2e}")
  ok &= rt < 1e-3

  # 3. Simulator: PD-track the reference joints open loop (no policy) from reference starts.
  n = sim.num_envs
  env = torch.arange(n, device=dev)
  motion = torch.randint(0, lib.num_motions, (n,), device=dev)
  frame = (torch.rand(n, device=dev) * (lib.length[motion] - 60).clamp(min=1)).long()
  gi = lib.index(motion, frame)
  sim.set_state(env, lib.qpos[gi], lib.qvel[gi])
  sim.randomize(env, torch.ones(n, device=dev), torch.zeros(n, 3, device=dev))
  xi = torch.zeros(n, XI_DIM, device=dev)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  t0 = time.time()
  for k in range(50):
    gi = lib.index(motion, frame + k + 1)
    ref_q = lib.qpos[gi, 7:]
    action = (ref_q[:, task.pid] - task.default_pos[task.pid]) / task.action_scale
    sim.step(task.joint_targets(action, xi))
    s = st.from_qpos_qvel(sim.qpos, sim.qvel)
    alive &= ~task.terminated(s, motion, frame + k + 1) & torch.isfinite(s).all(-1)
  dt = time.time() - t0
  print(f"open-loop PD replay, 1 s: {alive.float().mean().item() * 100:.1f}% of {n} envs not terminated; "
        f"{50 * n / dt:.0f} env-steps/s incl. Python overhead")
  ok &= bool(torch.isfinite(sim.qpos).all())
  print("CHECKS", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
  main()
