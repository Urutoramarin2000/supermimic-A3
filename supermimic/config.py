"""Hyperparameters. Defaults follow SuperMimic Table A.5 (one-GPU column); presets shrink them."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
  # Paths (relative ones are resolved against the repository root, see resolve()).
  xml: str = "assets/a3_racket/a3_pingpong.xml"
  motion_dir: str = "data/a3_31dof_paddle_50hz"
  motion_glob: str = "*.npz"
  out_dir: str = "runs/debug"
  resume: str = ""
  seed: int = 0
  device: str = "cuda:0"

  # Robot and simulator (MuJoCo Warp, 50 Hz control = 4 x 5 ms substeps).
  gains: str = "deploy"  # deploy: A3 deploy Kp/Kd, action scale 0.25. beyondmimic: armature-based 10 Hz, zeta 2.
  sim_dt: float = 0.005
  decimation: int = 4
  num_envs: int = 4096
  nconmax: int = 32
  njmax: int = 160
  solver_iterations: int = 10
  ls_iterations: int = 20

  # Observation (Table A.1) and goal (K future reference frames).
  hist_len: int = 10
  goal_frames: int = 10
  goal_root_pos: bool = True  # Include reference root position relative to the robot root in the goal.
  noise_joint_pos: float = 0.01
  noise_joint_vel: float = 0.5
  noise_ang_vel: float = 0.2
  noise_gravity: float = 0.05

  # Domain randomization = sim context xi (Table A.3).
  friction_lo: float = 0.3
  friction_hi: float = 1.2
  calib_range: float = 0.01
  com_x: float = 0.025
  com_y: float = 0.05
  com_z: float = 0.05
  action_noise: float = 0.1

  # Reward (Table A.2) and termination (Appendix A).
  w_body_pos: float = 1.0
  w_body_ori: float = 1.0
  w_body_linvel: float = 1.0
  w_body_angvel: float = 1.0
  w_root_pos: float = 0.5
  w_root_ori: float = 0.5
  w_racket_pos: float = 0.0  # pingpong extension: racket site position (0 = paper reward)
  w_racket_ori: float = 0.0
  std_body_pos: float = 0.3
  std_body_ori: float = 0.4
  std_body_linvel: float = 1.0
  std_body_angvel: float = 3.14
  std_root_pos: float = 0.3
  std_root_ori: float = 0.4
  std_racket_pos: float = 0.1
  std_racket_ori: float = 0.4
  w_action_rate: float = -0.1
  w_joint_limit: float = -2.0
  soft_limit_factor: float = 0.9
  term_height: float = 0.25
  term_tilt: float = 0.8
  soft_term_temp: float = 0.02  # sigmoid temperature of the differentiable termination in imagination

  # World model (Sec. 3.2, Table A.4/A.5).
  wm_hidden: tuple = (2048, 2048, 2048)
  wm_window: int = 8
  wm_batch: int = 2048
  wm_minibatches: int = 10

  # Policy / critic (Table A.4/A.5).
  enc_hidden: tuple = (2048, 1024, 512, 256)
  dec_hidden: tuple = (2048, 1024, 512, 256)
  latent_dim: int = 32
  critic_hidden: tuple = (256, 128)
  horizon: int = 32
  imag_batch: int = 4096
  gamma: float = 0.99
  lam: float = 0.95
  critic_epochs: int = 5
  critic_minibatches: int = 4

  # Optimization and schedule.
  lr: float = 2e-4
  lr_final_ratio: float = 0.03
  lr_decay_iters: int = 250_000
  max_iters: int = 320_000
  utd: float = 0.4  # policy updates per simulator step (per env)
  replay_len: int = 1000
  grad_clip: float = 1.0
  actor_elem_clip: float = 0.1
  warmup_collect_steps: int = 64
  wm_warmup_updates: int = 500

  # Engineering knobs (not in the paper).
  precision: str = "tf32"  # tf32 | bf16 (autocast for the networks)
  checkpoint_imagination: bool = False  # activation checkpointing per imagined step to save memory
  compile: bool = True  # torch.compile the imagined step and the world-model window loss (kernel fusion)
  log_every: int = 50
  save_every: int = 5000
  max_minutes: float = 0.0  # stop after this wall-clock budget (0 = no limit)

  # Return-based reset sampling (Appendix D.1).
  bin_seconds: float = 1.0
  reset_uniform: float = 0.5
  reset_hard: float = 0.4
  reset_past: float = 0.1
  reset_temp: float = 25.0
  bin_ema: float = 0.1


PRESETS = {
  # Paper defaults on one GPU.
  "paper": {},
  # Fits a shared 24 GB RTX 4090 with headroom: bf16 networks, smaller replay.
  "4090": {"precision": "bf16", "replay_len": 500},
  # Paper settings plus racket position/orientation rewards, 50k iterations (runs/racket_w1, ~12 h on a 4090).
  "racket": {"w_racket_pos": 1.0, "w_racket_ori": 1.0, "lr_decay_iters": 40_000, "max_iters": 50_000,
             "save_every": 2500},
  # Minutes-long functional test.
  "smoke": {
    "num_envs": 1024, "replay_len": 200, "imag_batch": 1024, "wm_batch": 1024, "wm_minibatches": 2,
    "warmup_collect_steps": 32, "wm_warmup_updates": 50, "max_iters": 300, "log_every": 10,
    "save_every": 150, "lr_decay_iters": 300,
  },
}


def resolve(path: str) -> str:
  """Expands ~ and resolves a relative path against the repository root (independent of the cwd)."""
  path = os.path.expanduser(path)
  return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _parse_value(text: str, default):
  if isinstance(default, bool):
    return text.lower() in ("1", "true", "yes", "y")
  if isinstance(default, tuple):
    return tuple(int(x) for x in text.split(","))
  return type(default)(text)


def get_config(argv=None) -> Config:
  parser = argparse.ArgumentParser()
  parser.add_argument("--preset", default="paper", help=f"one or more of {list(PRESETS)}, comma separated")
  defaults = Config()
  for f in dataclasses.fields(Config):
    parser.add_argument(f"--{f.name}", type=str, default=None)
  args = parser.parse_args(argv)
  values = {}
  for name in args.preset.split(","):
    values.update(PRESETS[name])
  for f in dataclasses.fields(Config):
    text = getattr(args, f.name)
    if text is not None:
      values[f.name] = _parse_value(text, getattr(defaults, f.name))
  return dataclasses.replace(defaults, **values)


def to_json(cfg: Config) -> str:
  return json.dumps(dataclasses.asdict(cfg), indent=2)
