"""Agibot A3 31-dof + right-hand paddle: joint tables, PD gains, MJCF build for MuJoCo Warp.

Gains, effort limits and the default pose are copied from smash_a3_native
(source/smash_sim/assets/robots/agibot.py), which matches the A3 deploy stack.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

JOINT_NAMES = [
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "head_yaw_joint", "head_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
NUM_JOINTS = len(JOINT_NAMES)  # 31, same order as the MJCF tree and the ChingMu npz qpos[7:]
HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")
# The policy drives 29 joints (head held at zero), like the deployed A3 policies.
POLICY_JOINT_IDS = [i for i, n in enumerate(JOINT_NAMES) if n not in HEAD_JOINTS]
NUM_ACTIONS = len(POLICY_JOINT_IDS)

ROOT_BODY = "pelvis_link"
TORSO_BODY = "torso_Link"
RACKET_SITE = "right_racket"
# BeyondMimic-style tracked bodies (smash_a3_native A3_TRACKED_BODIES).
TRACKED_BODIES = [
  "pelvis_link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
  "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link", "torso_Link",
  "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
  "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
]


def _by_pattern(table: dict[str, float]) -> np.ndarray:
  out = np.zeros(NUM_JOINTS)
  for i, name in enumerate(JOINT_NAMES):
    hits = [v for k, v in table.items() if k in name]
    assert len(hits) == 1, (name, hits)
    out[i] = hits[0]
  return out


KP = _by_pattern({
  "hip_yaw": 80.0, "hip_roll": 120.0, "hip_pitch": 80.0, "knee": 250.0, "ankle": 50.0,
  "waist_yaw": 85.0, "waist_roll": 50.0, "waist_pitch": 50.0, "head": 40.0,
  "shoulder_pitch": 40.0, "shoulder_roll": 40.0, "shoulder_yaw": 30.0, "elbow": 30.0,
  "wrist_roll": 30.0, "wrist_pitch": 20.0, "wrist_yaw": 20.0,
})
KD = _by_pattern({
  "hip_yaw": 3.0, "hip_roll": 4.0, "hip_pitch": 3.0, "knee": 8.0, "ankle": 2.0,
  "waist_yaw": 3.0, "waist_roll": 2.0, "waist_pitch": 2.0, "head": 2.0,
  "shoulder_pitch": 3.0, "shoulder_roll": 3.0, "shoulder_yaw": 2.0, "elbow": 2.0,
  "wrist_roll": 2.0, "wrist_pitch": 2.0, "wrist_yaw": 2.0,
})
EFFORT = _by_pattern({
  "hip_yaw": 220.0, "hip_roll": 220.0, "hip_pitch": 220.0, "knee": 320.0,
  "ankle_pitch": 118.2, "ankle_roll": 54.75,
  "waist_yaw": 220.0, "waist_roll": 46.0, "waist_pitch": 115.0, "head": 6.0,
  "shoulder_pitch": 60.0, "shoulder_roll": 60.0, "shoulder_yaw": 36.0, "elbow": 36.0,
  "wrist_roll": 36.0, "wrist_pitch": 6.0, "wrist_yaw": 6.0,
})
DEFAULT_POS = _by_pattern({
  "waist": 0.0, "head": 0.0,
  "shoulder_pitch": 0.20, "shoulder_yaw": 0.0, "elbow": 0.30, "wrist": 0.0,
  "hip_pitch": -0.15, "hip_roll": 0.0, "hip_yaw": 0.0, "knee": 0.30, "ankle_pitch": -0.15, "ankle_roll": 0.0,
  "left_shoulder_roll": 0.15, "right_shoulder_roll": -0.15,
})


def gains(kind: str, mjm: mujoco.MjModel | None = None):
  """Returns (kp, kd, action_scale) per joint."""
  if kind == "deploy":
    return KP.copy(), KD.copy(), np.full(NUM_JOINTS, 0.25)
  if kind == "beyondmimic":
    assert mjm is not None
    arm = np.array([mjm.dof_armature[mjm.jnt_dofadr[mjm.joint(n).id]] for n in JOINT_NAMES])
    omega, zeta = 2 * math.pi * 10.0, 2.0
    kp, kd = arm * omega**2, 2 * zeta * arm * omega
    return kp, kd, 0.25 * EFFORT / kp
  raise ValueError(kind)


def build_mjcf(xml_path: str, kind: str = "deploy", sim_dt: float = 0.005, iterations: int = 10,
               ls_iterations: int = 20) -> tuple[mujoco.MjModel, dict]:
  """Turns a3_pingpong.xml into a MuJoCo Warp friendly model.

  - torque motors -> PD position servos (general actuator, gain kp, bias -kp q - kd qd, force limits);
  - 5 ms implicitfast steps, 10 solver / 20 line-search iterations, no noslip (as in the SuperMimic demo);
  - robot-robot collisions off (like the Isaac A3 config); robot collision geoms get priority 1 so the
    per-env robot friction (domain randomization) decides the floor contact friction.
  """
  root = ET.parse(xml_path).getroot()
  compiler = root.find("compiler")
  compiler.set("meshdir", os.path.join(os.path.dirname(os.path.abspath(xml_path)), compiler.get("meshdir", "")))
  option = root.find("option")
  for key in ("noslip_iterations", "noslip_tolerance"):
    option.attrib.pop(key, None)
  option.set("timestep", str(sim_dt))
  option.set("integrator", "implicitfast")
  option.set("iterations", str(iterations))
  option.set("ls_iterations", str(ls_iterations))
  for default in root.iter("default"):
    if default.get("class") == "collision":
      geom = default.find("geom")
      geom.set("conaffinity", "0")
      geom.set("priority", "1")
  for tag in ("actuator", "sensor"):
    for node in root.findall(tag):
      root.remove(node)
  # Placeholder actuators; gains are written below once the model (and its armature) is known.
  act = ET.SubElement(root, "actuator")
  for name in JOINT_NAMES:
    ET.SubElement(act, "general", name=name, joint=name, ctrllimited="false", forcelimited="true",
                  forcerange="-1 1", biastype="affine", gainprm="1", biasprm="0 -1 -1")
  mjm = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  kp, kd, scale = gains(kind, mjm)
  for i, name in enumerate(JOINT_NAMES):
    a = mjm.actuator(name).id
    mjm.actuator_gainprm[a, 0] = kp[i]
    mjm.actuator_biasprm[a, :3] = (0.0, -kp[i], -kd[i])
    mjm.actuator_forcerange[a] = (-EFFORT[i], EFFORT[i])
  info = {
    "kp": kp, "kd": kd, "action_scale": scale, "default_pos": DEFAULT_POS.copy(),
    "joint_range": np.array([mjm.jnt_range[mjm.joint(n).id] for n in JOINT_NAMES]),
    "robot_geoms": np.array([g for g in range(mjm.ngeom) if mjm.geom_priority[g] == 1]),
    "torso_body": mjm.body(TORSO_BODY).id,
  }
  return mjm, info
