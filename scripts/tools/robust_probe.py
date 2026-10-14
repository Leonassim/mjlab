"""Falls under perturbation: 16 envs, 20 s, perturbations of RHPS1_ROBUST active.

  RHPS1_ROBUST=push RHPS1_PUSH_SCALE=0.4 uv run python scripts/tools/robust_probe.py <run> <ckpt>

Each env holds one command from rest (standing, forward, lateral left, yaw left,
backward); a fall is the pelvis below 0.55 m. The command reaches the policy
through the controller's filter (RHPS1_CMD_TAU), as on the robot.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
CMDS = {"debout": (0, 0, 0), "avant": (1, 0, 0), "gauche": (0, 1, 0), "rot+": (0, 0, 1), "arriere": (-1, 0, 0)}
PER = 16
N = PER * len(CMDS)
torch.manual_seed(int(os.environ.get("PROBE_SEED", "0")))
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.terminations = {}
cfg.episode_length_s = 1e9
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
# No reference resets mid-run: every env starts at frame 0 of the first clip
# (long enough for 23 s), as in play. Otherwise RSI and the clip seam teleport
# envs and fake both falls and drift.
_m = cfg.commands["motion"]
_m.sampling_mode = "start"
_m.pose_range = {}
_m.velocity_range = {}
if hasattr(_m, "standing_start_prob"):
  _m.standing_start_prob = 0.0
_lag = int(round(float(os.environ.get("FLIP_DELAY_MS", "30")) / 1000 / cfg.sim.mujoco.timestep))
for _a in cfg.scene.entities["robot"].articulation.actuators:
  _a.delay_min_lag = _lag
  _a.delay_max_lag = _lag
TARGET = torch.tensor([c for c in CMDS.values() for _ in range(PER)], dtype=torch.float32)
CMD = torch.zeros(N, 3)
TAU = float(os.environ.get("RHPS1_CMD_TAU", "0.3"))
_filt = {"y": torch.zeros(N, 3), "step": -1}


def _cmd(env, command_name, tau=0.0):
  step = int(env.common_step_counter)
  if _filt["step"] != step:
    _filt["y"] = _filt["y"] + (env.step_dt / TAU) * (CMD - _filt["y"])
    _filt["step"] = step
  return _filt["y"].to(env.device)


cfg.observations["actor"].terms["velocity_command"].func = _cmd
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True},
            strict=True, map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
robot = u.scene["robot"]
dt = u.step_dt
obs, _ = env.reset()
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(q0 + (torch.rand_like(q0) * 2 - 1) * 0.03, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)
fell = torch.zeros(N, dtype=torch.bool)
start = robot.data.root_link_pos_w[:, :2].clone()


def _yaw():
  q = robot.data.root_link_quat_w
  return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


yaw0 = _yaw().clone()
yaw_acc = torch.zeros(N)
yaw_prev = yaw0.clone()
with torch.no_grad():
  for k in range(int(23.0 / dt)):
    CMD[:] = TARGET if k * dt >= 3.0 else 0.0
    obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
    fell |= robot.data.root_link_pos_w[:, 2] < 0.55
    y = _yaw()
    yaw_acc += torch.atan2(torch.sin(y - yaw_prev), torch.cos(y - yaw_prev))
    yaw_prev = y
d = (robot.data.root_link_pos_w[:, :2] - start).numpy()
print(f"{run} {ck}  RHPS1_ROBUST={os.environ.get('RHPS1_ROBUST', '')} push x{os.environ.get('RHPS1_PUSH_SCALE', '-')}"
      f" friction {os.environ.get('RHPS1_FRICTION_RANGE', '-')}  retard {_lag * cfg.sim.mujoco.timestep * 1000:.0f} ms")
for i, (name, c) in enumerate(CMDS.items()):
  s = slice(i * PER, (i + 1) * PER)
  print(f"  {name:8s} chutes {int(fell[s].sum()):2d}/{PER}   deplacement median dx {np.median(d[s, 0]):+.2f} dy {np.median(d[s, 1]):+.2f} m   lacet median {np.degrees(np.median(yaw_acc[s].numpy())):+5.0f} deg (p10 {np.degrees(np.percentile(yaw_acc[s].numpy(), 10)):+.0f} p90 {np.degrees(np.percentile(yaw_acc[s].numpy(), 90)):+.0f})")
