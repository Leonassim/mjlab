"""Yaw torque pulse on the pelvis, joints held: frequency and damping of the free torsion.

  <plant env vars> uv run python scripts/tools/yaw_push_sim.py [torque_Nm] [duration_s]

Robot (decay after disarming slot 11, 2026-10-23): 5.3-5.9 Hz, zeta ~0.05. Nominal sim
(hip-yaw spring 600, damping 0.3-1.0 x PD): no free oscillation. K 2200 with dampings 0.001:
5.36 Hz, zeta 0.09 (the floor left by the contact).
"""

import sys

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

TQ = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
cfg = load_env_cfg("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start", play=True)
cfg.scene.num_envs = 1
cfg.terminations = {}
cfg.episode_length_s = 1e9
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
m = cfg.commands["motion"]
m.sampling_mode = "start"
m.pose_range = {}
m.velocity_range = {}
if hasattr(m, "standing_start_prob"):
  m.standing_start_prob = 0.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
robot = u.scene["robot"]
body = [list(robot.body_names).index("BODY")]
env.reset()
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)
dt = u.step_dt
zero = torch.zeros(1, env.num_actions)
t, w = [], []
with torch.no_grad():
  for k in range(int(8.0 / dt)):
    tt = k * dt
    tq = torch.zeros(1, 1, 3)
    if 3.0 <= tt < 3.0 + DUR:
      tq[0, 0, 2] = TQ
    robot.write_external_wrench_to_sim(torch.zeros(1, 1, 3), tq, body_ids=body)
    env.step(zero)
    t.append(tt)
    w.append(float(np.degrees(robot.data.root_link_ang_vel_b[0, 2])))
t, w = np.array(t), np.array(w)
fs = 1 / dt
b, a = butter(2, [2.0 / (fs / 2), 15.0 / (fs / 2)], "band")
x = filtfilt(b, a, w)
mm = t > 3.0 + DUR + 0.02
pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0))[0] + 1
pk = pk[mm[pk]]
pk = pk[x[pk] > 0.03 * x[pk].max()] if len(pk) else pk
if len(pk) < 3:
  print(
    f"couple {TQ} N.m x {DUR} s : pas d'oscillation libre (pics {len(pk)}), lacet max {np.abs(w[mm]).max():.2f} deg/s"
  )
  sys.exit()
A = x[pk][:8]
T = np.diff(t[pk][:8]).mean()
d = np.log(A[0] / A[-1]) / (len(A) - 1)
print(
  f"couple {TQ} N.m x {DUR} s : {len(pk)} pics, {1 / T:.2f} Hz, {A[0]:.2f} -> {A[-1]:.2f} deg/s, zeta {d / np.sqrt(4 * np.pi**2 + d**2):.3f}"
)
