"""Deterministic rollout under training conditions on a one-clip motion file: does the policy walk?

  <training env vars> uv run python scripts/tools/clip_follow.py <run> <ckpt> <one_clip.npz>

Per step, robot vs reference forward speed while the command says forward / backward
(resets excluded). The push probe scores a robot that stands as robust; this does not.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

run, ck, motion = sys.argv[1], sys.argv[2], sys.argv[3]
N = 16
cfg = load_env_cfg(g.TASK, play=False)
cfg.scene.num_envs = N
cfg.commands["motion"].motion_file = motion
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
runner = load_runner_cls(g.TASK)(env, asdict(load_rl_cfg(g.TASK)), device="cpu")
runner.load(
  str(Path("logs/rsl_rl/rhps1_tracking") / run / ck),
  load_cfg={"actor": True},
  strict=True,
  map_location="cpu",
)
pol = runner.get_inference_policy(device="cpu")
robot = u.scene["robot"]
cmd = u.command_manager.get_term("motion")
obs, _ = env.reset()
vr, vf, cx, ok = [], [], [], []
prev_r = robot.data.root_link_pos_w[:, :2].clone()
prev_f = cmd.anchor_pos_w[:, :2].clone()
with torch.no_grad():
  for _ in range(int(12.0 / u.step_dt)):
    o = obs["actor"] if hasattr(obs, "keys") else obs
    cx.append(o[:, -3].numpy().copy())  # newest command vx, last term in every format
    obs, _, d, _ = env.step(pol(obs))
    r, f = robot.data.root_link_pos_w[:, :2].clone(), cmd.anchor_pos_w[:, :2].clone()
    vr.append(((r - prev_r)[:, 0] / u.step_dt).numpy())
    vf.append(((f - prev_f)[:, 0] / u.step_dt).numpy())
    ok.append(~d.bool().numpy())
    prev_r, prev_f = r, f
vr, vf, cx, ok = map(np.array, (vr, vf, cx, ok))
for lab, m in (
  ("commande avant", (cx > 0.5) & ok),
  ("commande arriere", (cx < -0.5) & ok),
  ("commande nulle", (np.abs(cx) < 0.1) & ok),
):
  if m.sum():
    print(
      f"{motion.split('/')[-1]} {lab:16s}: {m.mean() * 100:4.0f}% du temps | robot vx {vr[m].mean():+.3f} m/s | reference vx {vf[m].mean():+.3f} m/s"
    )
