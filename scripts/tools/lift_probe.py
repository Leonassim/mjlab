"""Foot lift per command direction: policy in mjlab against the BWC reference clips.

  uv run python scripts/tools/lift_probe.py <run> <ckpt>

Lift = per swing, the ankle's peak height above its own stance height (swings
are excursions above stance + 1 cm). The reference columns read the v3 clips
(forward, left, backward, right, yaw left, yaw right) with the same rule.
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
MOTION = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
CMDS = {"avant": (1, 0, 0), "gauche": (0, 1, 0), "arriere": (-1, 0, 0), "droite": (0, -1, 0),
        "rot+": (0, 0, 1), "rot-": (0, 0, -1)}
PER = 8


def lifts(z, dt):
  """Peak above stance per swing, z [T]."""
  stance = np.percentile(z, 5)
  up = z > stance + 0.01
  e = np.diff(np.r_[0, up.astype(int), 0])
  on, off = np.where(e == 1)[0], np.where(e == -1)[0]
  return np.array([z[a:b].max() - stance for a, b in zip(on, off) if (b - a) * dt > 0.15])


cfg = load_env_cfg(TASK, play=True)
N = PER * len(CMDS)
cfg.scene.num_envs = N
for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
  cfg.terminations.pop(k, None)
cfg.commands["motion"].motion_file = MOTION
_lag = int(round(30 / 1000 / cfg.sim.mujoco.timestep))
for _a in cfg.scene.entities["robot"].articulation.actuators:
  _a.delay_min_lag = _lag
  _a.delay_max_lag = _lag
TARGET = torch.tensor([c for c in CMDS.values() for _ in range(PER)], dtype=torch.float32)
CMD = torch.zeros(N, 3)
_filt = {"y": torch.zeros(N, 3), "step": -1}


def _cmd(env, command_name, tau=0.0):
  step = int(env.common_step_counter)
  if _filt["step"] != step:
    _filt["y"] = _filt["y"] + (env.step_dt / 0.3) * (CMD - _filt["y"])
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
feet = [robot.body_names.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
dt = u.step_dt
obs, _ = env.reset()
Z = []
with torch.no_grad():
  for k in range(int(18.0 / dt)):
    CMD[:] = TARGET if k * dt >= 3.0 else 0.0
    obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
    if k * dt >= 6.0:
      Z.append(robot.data.body_link_pos_w[:, feet, 2].numpy().copy())
Z = np.stack(Z)

# Reference: same rule on each v3 clip, walking part only (ankle above stance).
m = np.load(MOTION)
starts = list(m["clip_starts"]) + [len(m["joint_pos"])]
ref_names = ["avant", "gauche", "arriere", "droite", "rot+", "rot-"]
bi = [list(robot.body_names).index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
fps = float(m["fps"][0])
ref = {}
for name, a, b in zip(ref_names, starts[:-1], starts[1:]):
  zz = m["body_pos_w"][a:b][:, bi, 2]
  ref[name] = [lifts(zz[:, j], 1 / fps) for j in range(2)]

print(f"{run} {ck}  levee de cheville par pas (cm), mediane G / D, retard 30 ms")
print(f"  {'commande':8s} {'policy':>13s} {'BWC':>13s}   pas policy")
for i, name in enumerate(CMDS):
  s = slice(i * PER, (i + 1) * PER)
  pl = [np.concatenate([lifts(Z[:, e, j], dt) for e in range(s.start, s.stop)]) for j in range(2)]
  pm = [np.median(x) * 100 if len(x) else 0 for x in pl]
  rm = [np.median(x) * 100 if len(x) else 0 for x in ref[name]]
  print(f"  {name:8s} {pm[0]:5.1f} / {pm[1]:4.1f}   {rm[0]:5.1f} / {rm[1]:4.1f}   {len(pl[0])//PER}/{len(pl[1])//PER} par env")
