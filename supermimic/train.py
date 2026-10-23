"""SuperMimic training loop (Algorithm A.1) for Agibot A3 on the ChingMu table-tennis motions.

  python -m supermimic.train --preset paper --out_dir runs/a3_chingmu
  torchrun --nproc_per_node 2 -m supermimic.train ...   # data parallel, gradients averaged
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .agent import SuperMimic
from .config import get_config, resolve, to_json
from .env import Collector
from .kinematics import Kinematics
from .motion import MotionLibrary
from .replay import Replay
from .robot import NUM_ACTIONS, RACKET_SITE, TRACKED_BODIES, build_mjcf
from .sampler import ResetSampler
from .sim import WarpSim
from .task import XI_DIM, TrackingTask


def build(cfg, device, num_envs=None, with_sim=True):
  mjm, info = build_mjcf(resolve(cfg.xml), cfg.gains, cfg.sim_dt, cfg.solver_iterations, cfg.ls_iterations)
  kin = Kinematics(mjm, TRACKED_BODIES, [RACKET_SITE], device)
  lib = MotionLibrary(resolve(cfg.motion_dir), cfg.motion_glob, kin, device, cfg.bin_seconds)
  task = TrackingTask(cfg, kin, lib, info, device)
  sim = None
  if with_sim:
    sim = WarpSim(mjm, info, num_envs or cfg.num_envs, cfg.decimation, str(device), cfg.nconmax, cfg.njmax)
  return mjm, info, kin, lib, task, sim


def main():
  cfg = get_config()
  distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
  rank = 0
  if distributed:
    # nccl across GPUs; SM_DIST_BACKEND=gloo lets several ranks share one GPU (testing the multi-GPU path).
    dist.init_process_group(os.environ.get("SM_DIST_BACKEND", "nccl"))
    rank = dist.get_rank()
    cfg.device = f"cuda:{int(os.environ['LOCAL_RANK']) % torch.cuda.device_count()}"
  device = torch.device(cfg.device)
  torch.cuda.set_device(device)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(cfg.seed + rank)
  np.random.seed(cfg.seed + rank)
  if rank == 0:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
      f.write(to_json(cfg))

  t0 = time.time()
  mjm, info, kin, lib, task, sim = build(cfg, device)
  sampler = ResetSampler(lib, cfg, task.max_step_reward)
  collector = Collector(cfg, sim, task, lib, sampler, device)
  replay = Replay(cfg.replay_len, cfg.num_envs, NUM_ACTIONS, XI_DIM, device)
  agent = SuperMimic(cfg, task, device)
  start_it = 0
  if cfg.resume:
    ck = torch.load(cfg.resume, map_location=device, weights_only=False)
    agent.load_state_dict(ck["agent"])
    sampler.load_state_dict(ck["sampler"])
    start_it = ck["iter"]
  if rank == 0:
    print(f"[setup] {lib.num_motions} motions, {lib.seconds / 60:.1f} min, {lib.num_bins} bins; "
          f"{cfg.num_envs} envs; goal {task.goal_dim}d; setup {time.time() - t0:.1f}s", flush=True)
    n_params = {k: sum(p.numel() for p in getattr(agent, k).parameters()) for k in ("wm", "actor", "critic")}
    print(f"[setup] params {n_params}", flush=True)

  # Warm-up: fill the replay with the initial (near-zero) policy plus action noise, pre-fit the world model.
  def update_normalizers():
    with torch.no_grad():
      agent.actor.obs_norm.update(collector.hist)
      agent.actor.goal_norm.update(task.goal(collector.state(), collector.motion, collector.frame))

  for _ in range(cfg.warmup_collect_steps):
    collector.step(agent.actor, replay)
    update_normalizers()
  wm_logs = {}
  for _ in range(cfg.wm_warmup_updates // max(cfg.wm_minibatches, 1)):
    wm_logs = agent.wm_update(replay)
  if rank == 0:
    print(f"[warmup] {cfg.warmup_collect_steps} steps, wm {wm_logs}, {time.time() - t0:.0f}s", flush=True)
  collector.pop_stats()

  log_path = os.path.join(cfg.out_dir, "log.jsonl")
  debt, t_start = 0.0, time.time()
  timers = {"collect": 0.0, "wm": 0.0, "policy": 0.0}
  acc: dict[str, float] = {}
  for it in range(start_it, cfg.max_iters):
    lr = agent.set_lr(it)
    tic = time.time()
    debt += 1.0 / cfg.utd
    while debt >= 1.0:
      collector.step(agent.actor, replay)
      debt -= 1.0
    torch.cuda.synchronize(device)
    timers["collect"] += time.time() - tic
    tic = time.time()
    logs = agent.wm_update(replay)
    torch.cuda.synchronize(device)
    timers["wm"] += time.time() - tic
    tic = time.time()
    logs.update(agent.policy_update(replay))
    update_normalizers()  # observation/goal statistics follow simulator data, fixed inside gradient passes
    torch.cuda.synchronize(device)
    timers["policy"] += time.time() - tic
    for k, v in logs.items():
      acc[k] = acc.get(k, 0.0) + v
    done_its = it + 1 - start_it
    if rank == 0 and (it + 1) % cfg.log_every == 0:
      row = {k: v / cfg.log_every for k, v in acc.items()}
      row.update(collector.pop_stats())
      el = time.time() - t_start
      row.update({"iter": it + 1, "lr": lr, "minutes": el / 60, "it_per_s": done_its / el,
                  "env_steps": replay.t * cfg.num_envs,
                  "mem_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                  **{f"t_{k}": v / done_its for k, v in timers.items()}})
      acc = {}
      with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")
      print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    stop = bool(cfg.max_minutes) and (time.time() - t_start) / 60 > cfg.max_minutes
    if distributed:  # every rank must take the same decision
      flag = torch.tensor(float(stop), device=device)
      dist.all_reduce(flag, op=dist.ReduceOp.MAX)
      stop = bool(flag.item())
    if rank == 0 and ((it + 1) % cfg.save_every == 0 or it + 1 == cfg.max_iters or stop):
      path = os.path.join(cfg.out_dir, f"model_{it + 1}.pt")
      torch.save({"agent": agent.state_dict(), "sampler": sampler.state_dict(), "iter": it + 1,
                  "config": json.loads(to_json(cfg))}, path)
      print(f"[save] {path}", flush=True)
    if stop:
      break
  if distributed:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
