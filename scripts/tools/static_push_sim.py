"""Static push test in sim, as on the robot (2026-10-22): joints held, one push on the pelvis,
then the free pitch oscillation's frequency and damping (log decrement of the peaks).

  <env vars of the plant> uv run python scripts/tools/static_push_sim.py [force_N] [duration_s] [run ckpt]

With run and ckpt the policy holds the robot instead (standing command, armed at 1 s).
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

F = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
cfg = load_env_cfg("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start", play=True)
cfg.scene.num_envs = 1
cfg.terminations = {}
cfg.episode_length_s = 1e9
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
m = cfg.commands["motion"]
m.sampling_mode = "start"
m.pose_range = {}
m.velocity_range = {}
if hasattr(m, "standing_start_prob"):
  m.standing_start_prob = 0.0
POLICY = sys.argv[3:5] if len(sys.argv) > 4 else None
if POLICY:
  cfg.observations["actor"].terms["velocity_command"].func = lambda env, command_name, tau=0.0: torch.zeros(
    env.num_envs, 3, device=env.device)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
robot = u.scene["robot"]
body = [list(robot.body_names).index("BODY")]
obs, _ = env.reset()
pol = None
if POLICY:
  runner = load_runner_cls("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start")(
    env, asdict(load_rl_cfg("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start")), device="cpu")
  runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / POLICY[0] / POLICY[1]), load_cfg={"actor": True},
              strict=True, map_location="cpu")
  pol = runner.get_inference_policy(device="cpu")
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)
dt = u.step_dt
t, pitch = [], []
zero = torch.zeros(1, env.num_actions)
with torch.no_grad():
  for k in range(int(14.0 / dt)):
    tt = k * dt
    f = torch.zeros(1, 1, 3)
    if 3.0 <= tt < 3.0 + DUR:
      f[0, 0, 0] = F
    robot.write_external_wrench_to_sim(f, torch.zeros(1, 1, 3), body_ids=body)
    obs, _, _, _ = env.step(pol(obs) if pol and tt >= 1.0 else zero)
    g = robot.data.projected_gravity_b[0]
    t.append(tt)
    pitch.append(float(torch.rad2deg(torch.atan2(-g[0], -g[2]))))
t, pitch = np.array(t), np.array(pitch)
fs = 1 / dt
b, a = butter(2, [0.4 / (fs / 2), 6.0 / (fs / 2)], btype="band")
pf = filtfilt(b, a, pitch)
m_ = t > 3.0 + DUR + 0.2
pk = np.where((pf[1:-1] > pf[:-2]) & (pf[1:-1] > pf[2:]))[0] + 1
pk = pk[m_[pk]]
pk = pk[pf[pk] > 0.02 * np.abs(pf[m_]).max()]
if len(pk) < 3:
  print(f"poussee {F:.0f} N x {DUR} s : pas d'oscillation libre (pics {len(pk)}), tangage ptp {np.ptp(pitch[m_]):.3f} deg")
  sys.exit()
T = np.diff(t[pk]).mean()
A = pf[pk][:6]
delta = np.log(A[0] / A[-1]) / (len(A) - 1)
zeta = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)
print(f"poussee {F:.0f} N x {DUR} s : {len(pk)} pics, periode {T:.3f} s ({1 / T:.2f} Hz), amplitude {A[0]:.2f} -> {A[-1]:.2f} deg, zeta {zeta:.3f}")
