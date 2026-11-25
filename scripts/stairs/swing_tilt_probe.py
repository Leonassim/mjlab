"""Swing sole tilt and swing knee under a forward command (oseq), against the BWC swing-foot envelope.

  .venv/bin/python scripts/stairs/swing_tilt_probe.py <run> <model_XXXX.pt> [vx,vy,wz]

Run with the recipe's env (stairs_env <recipe>). Prints p50/p90/p99 of |sole pitch| and |sole roll| while the foot is
unloaded, the share of swing time beyond the envelope (pitch 22.3, roll 9.5 deg), and the swing-knee flexion above
the stance knee.
"""

import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ["RHPS1_STAIRS_OSEQ_P"] = "1"
for k in ("RHPS1_STAIRS_FRAC", "RHPS1_STAIRS_STAND_P", "RHPS1_STAIRS_PAUSE_P", "RHPS1_STAIRS_STAIR_PAUSE_P"):
  os.environ[k] = "0"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1.stairs import BWC_SWING_PITCH, BWC_SWING_ROLL, _foot_tilt  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
cmd = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,0,0").split(",")]
N, dev = 16, "cuda:0"
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
u = env.unwrapped
st = u._stairs
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True}, strict=True,
            map_location=dev)
pol = runner.get_inference_policy(device=dev)
u.reset()
obs = env.get_observations()
st.oseq[:] = True
st.seg_cmd[:] = 0.0
st.seg_cmd[:, 0] = torch.tensor(cmd, device=dev)
st.seg_end[:] = 10**9
robot = u.scene["robot"]
kid = [robot.joint_names.index(n) for n in ("L_KNEE_P", "R_KNEE_P")]
alive = torch.ones(N, dtype=torch.bool, device=dev)
pit_s, rol_s, kex_s = [], [], []
BW = 57.64 * 9.81
one_t, all_t = 0.0, 0.0
lift_pk = torch.zeros(N, 2, device=dev)
lifts = []
for i in range(int(8 / u.step_dt)):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  alive &= ~u.termination_manager.terminated
  if i * u.step_dt < 2.0:
    continue
  pit, rol = _foot_tilt(u)
  sw = (st.load <= 10.0) & alive[:, None]
  q = robot.data.joint_pos[:, kid]
  kex = torch.stack([q[:, 0] - q[:, 1], q[:, 1] - q[:, 0]], -1)
  ld = st.load > 0.1 * BW
  one_t += float(((ld.sum(-1) == 1) & alive).sum())
  all_t += float(alive.sum())
  h = robot.data.body_link_pos_w[:, st.foot_ids, 2] - 0.10 - u.scene["stairs_ground"].data.hit_pos_w[:, :2, 2]
  lift_pk = torch.where(sw, torch.maximum(lift_pk, h), lift_pk)
  landed = st.land & alive[:, None]
  lifts.append(lift_pk[landed].cpu())
  lift_pk = torch.where(sw, lift_pk, torch.zeros_like(lift_pk))
  pit_s.append(pit.abs()[sw].cpu())
  rol_s.append(rol.abs()[sw].cpu())
  kex_s.append(kex[sw].cpu())
pit_a, rol_a, kex_a = (torch.cat(x).numpy() for x in (pit_s, rol_s, kex_s))
d = np.degrees
print(f"SWINGTILT {run}/{ck} cmd {cmd}: {len(pit_a)} swing samples, {int(alive.sum())}/{N} alive")
print(f"  |pitch| p50/p90/p99 {d(np.percentile(pit_a, 50)):.1f}/{d(np.percentile(pit_a, 90)):.1f}/"
      f"{d(np.percentile(pit_a, 99)):.1f} deg, beyond {math.degrees(BWC_SWING_PITCH):.1f}: "
      f"{100 * np.mean(pit_a > BWC_SWING_PITCH):.1f} %")
print(f"  |roll|  p50/p90/p99 {d(np.percentile(rol_a, 50)):.1f}/{d(np.percentile(rol_a, 90)):.1f}/"
      f"{d(np.percentile(rol_a, 99)):.1f} deg, beyond {math.degrees(BWC_SWING_ROLL):.1f}: "
      f"{100 * np.mean(rol_a > BWC_SWING_ROLL):.1f} %")
lf = torch.cat(lifts).numpy() if lifts else np.zeros(1)
print(f"  single support {100 * one_t / max(all_t, 1):.0f} % | peak lift per step med {100 * np.median(lf):.1f} cm"
      f" ({len(lf)} steps)")
print(f"  swing knee over stance p50/p90/p99 {d(np.percentile(kex_a, 50)):+.1f}/{d(np.percentile(kex_a, 90)):+.1f}/"
      f"{d(np.percentile(kex_a, 99)):+.1f} deg")
