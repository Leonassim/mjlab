"""Standing vibration: policy armed at command 0 for 20 s, pelvis yaw rate 3.5-8 Hz rms per 2 s.

  <plant env vars> uv run python scripts/tools/rest_vibration.py <run> <ckpt>

On the robot (slot 11, 2026-10-23) the pelvis torsion grows 1.8 -> 13 deg/s at 5.5 Hz and
saturates; slot 3 stays ~7. Reproduced only with the identified torsion mode
(RHPS1_SERIES_K_CROTCH_Y=2200, DAMP_CROTCH_Y and DAMP_ANKLE_R 0.001) and RHPS1_ROBUST=obs,bias.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, welch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
N = 8
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.terminations = {}
cfg.episode_length_s = 1e9
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
m = cfg.commands["motion"]
m.sampling_mode = "start"
m.pose_range = {}
m.velocity_range = {}
if hasattr(m, "standing_start_prob"):
  m.standing_start_prob = 0.0
lag = int(
  round(float(os.environ.get("FLIP_DELAY_MS", "30")) / 1000 / cfg.sim.mujoco.timestep)
)
for a in cfg.scene.entities["robot"].articulation.actuators:
  a.delay_min_lag = lag
  a.delay_max_lag = lag
cfg.observations["actor"].terms["velocity_command"].func = (
  lambda env, command_name, tau=0.0: torch.zeros(env.num_envs, 3)
)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
robot = u.scene["robot"]
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(
  str(Path("logs/rsl_rl/rhps1_tracking") / run / ck),
  load_cfg={"actor": True},
  strict=True,
  map_location="cpu",
)
pol = runner.get_inference_policy(device="cpu")
torch.manual_seed(0)
obs, _ = env.reset()
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(
  q0 + (torch.rand_like(q0) * 2 - 1) * 0.03, torch.zeros_like(q0)
)
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)
dt = u.step_dt
W = []
with torch.no_grad():
  for k in range(int(22.0 / dt)):
    obs, *_ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
    W.append(np.degrees(robot.data.root_link_ang_vel_b[:, 2].numpy().copy()))
W = np.array(W)
fs = 1 / dt
b, a = butter(2, [3.5 / (fs / 2), 8 / (fs / 2)], "band")
Wf = filtfilt(b, a, W, axis=0)
s = int(2.0 / dt)
n = int(2.0 / dt)
ch = [np.median(Wf[i : i + n].std(0)) for i in range(s, len(W) - n + 1, n)]
f, P = welch(W[s:] - W[s:].mean(0), fs=fs, nperseg=1024, axis=0)
P = P.mean(1)
k = (f > 1) & (f < 30)
print(
  f"{run} {ck}: lacet 3.5-8 Hz rms par 2 s (mediane 8 envs): "
  + " ".join(f"{x:.2f}" for x in ch)
  + f" deg/s | pic {f[k][P[k].argmax()]:.1f} Hz"
)
