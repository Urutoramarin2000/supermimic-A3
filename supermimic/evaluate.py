"""Benchmark protocol of Appendix F.1 on the training motions: every motion plays once from its first frame,
friction 1.0, no randomization, no observation or action noise. A motion fails if the training termination
triggers. Also reports racket (paddle) tracking at the annotated hit frames.

  python -m supermimic.evaluate --ckpt runs/x/model_1000.pt [--device cuda:0]
  # checkpoint from another machine: point at the local robot model / motions instead of the trained paths
  python -m supermimic.evaluate --ckpt model.pt --xml assets/a3_racket/a3_pingpong.xml --motion_dir data/a3_31dof_paddle_50hz
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os

import torch

from . import state as st
from .agent import SuperMimic
from .config import Config
from .env import Collector
from .sampler import ResetSampler
from .sim import WarpSim
from .train import build


@torch.no_grad()
def evaluate(cfg, agent, lib, task, sim, device, motions=None, on_step=None):
  """motions: library ids to play, one per sim env (default: all). on_step(idx, racket_err, active) is called
  after every control step with the reference frame index of each env (lets callers record rollouts)."""
  motions = torch.arange(lib.num_motions, device=device) if motions is None else motions.to(device)
  n = len(motions)
  length = lib.length[motions]
  sampler = ResetSampler(lib, cfg, task.max_step_reward)
  eval_cfg = dataclasses.replace(cfg, noise_joint_pos=0.0, noise_joint_vel=0.0, noise_ang_vel=0.0,
                                 noise_gravity=0.0)
  task.noise_scale.zero_()
  col = Collector(eval_cfg, sim, task, lib, sampler, device, randomize=False)
  ids = torch.arange(n, device=device)
  col.reset(ids, motion=motions, frame=torch.zeros_like(ids))
  active = torch.ones(n, dtype=torch.bool, device=device)
  failed = torch.zeros(n, dtype=torch.bool, device=device)
  ret = torch.zeros(n, device=device)
  pos_err = torch.zeros(n, device=device)
  steps = torch.zeros(n, device=device)
  racket_err, racket_n = torch.zeros(n, device=device), torch.zeros(n, device=device)
  hit_mask = torch.zeros(lib.num_frames, dtype=torch.bool, device=device)
  for m, hits in enumerate(lib.hit_frames):
    if len(hits):
      hit_mask[lib.start[m] + torch.as_tensor(hits, device=device)] = True
  for _ in range(int(length.max())):
    s = col.state()
    goal = task.goal(s, col.motion, col.frame)
    action = agent.actor(col.hist, goal)
    sim.step(task.joint_targets(action, col.xi))
    s1 = col.state()
    r, _, body_err = task.reward(s1, col.motion, col.frame + 1, action, col.prev_action, return_terms=True)
    col.frame += 1
    idx = lib.index(col.motion, col.frame)
    bp = task.kin(*st.split(s1)[:3])[0]
    e_r = (bp[:, -1] - lib.body_pos[idx, -1]).norm(dim=-1)
    on_hit = hit_mask[idx] & active
    racket_err += torch.where(on_hit, e_r, 0.0)
    racket_n += on_hit.float()
    ret += torch.where(active, r, 0.0)
    pos_err += torch.where(active, body_err, 0.0)
    steps += active.float()
    if on_step is not None:
      on_step(idx, e_r, active)
    term = task.terminated(s1, col.motion, col.frame) | ~torch.isfinite(s1).all(-1)
    failed |= term & active
    active &= ~term & ~lib.ended(col.motion, col.frame)
    col.prev_action = action
    o = task.obs(s1, action, col.xi)
    col.hist = torch.cat([col.hist[:, 1:], o[:, None]], 1)
    if not active.any():
      break
  norm_ret = ret / (task.max_step_reward * (length - 1).float())
  rows = []
  for m in range(n):
    rows.append({
      "motion": lib.names[int(motions[m])], "frames": int(length[m]), "failed": bool(failed[m]),
      "norm_return": float(norm_ret[m]), "mean_body_pos_err_m": float(pos_err[m] / steps[m].clamp(min=1)),
      "hits": int(racket_n[m]),
      "racket_hit_err_m": float(racket_err[m] / racket_n[m]) if racket_n[m] > 0 else None,
    })
  hit_total = racket_n.sum().clamp(min=1)
  summary = {
    "motions": n, "failed": int(failed.sum()), "norm_return_mean": float(norm_ret.mean()),
    "mean_body_pos_err_m": float((pos_err / steps.clamp(min=1)).mean()),
    "hits_evaluated": int(racket_n.sum()), "racket_hit_err_m": float(racket_err.sum() / hit_total),
  }
  return summary, rows


def load(ckpt_path, device):
  """Config (as trained) and raw checkpoint dict."""
  ck = torch.load(ckpt_path, map_location=device, weights_only=False)
  fields = {f.name for f in dataclasses.fields(Config)}
  cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"].items() if k in fields})
  return dataclasses.replace(cfg, device=str(device)), ck


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", required=True)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default="")
  p.add_argument("--xml", default="", help="override the robot MJCF stored in the checkpoint config")
  p.add_argument("--motion_dir", default="", help="override the motion folder stored in the checkpoint config")
  a = p.parse_args()
  device = torch.device(a.device)
  torch.cuda.set_device(device)
  cfg, ck = load(a.ckpt, device)
  cfg = dataclasses.replace(cfg, xml=a.xml or cfg.xml, motion_dir=a.motion_dir or cfg.motion_dir)
  mjm, info, _, lib, task, _ = build(cfg, device, with_sim=False)
  sim = WarpSim(mjm, info, lib.num_motions, cfg.decimation, str(device), cfg.nconmax, cfg.njmax)
  agent = SuperMimic(cfg, task, device)
  agent.load_state_dict(ck["agent"])
  summary, rows = evaluate(cfg, agent, lib, task, sim, device)
  summary["ckpt"], summary["iter"] = a.ckpt, ck["iter"]
  print(json.dumps(summary, indent=2))
  out = a.out or os.path.splitext(a.ckpt)[0] + "_eval"
  with open(out + ".json", "w") as f:
    json.dump(summary, f, indent=2)
  with open(out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print(f"[eval] wrote {out}.json / .csv")


if __name__ == "__main__":
  main()
