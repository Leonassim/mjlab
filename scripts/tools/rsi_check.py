"""Tracks the reference once started mid-walk? Robot vs reference displacement.

  uv run python scripts/tools/rsi_check.py <run> <model.pt> [--secs 6]
"""
import argparse, os
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("checkpoint")
p.add_argument("--task", default="Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start")
p.add_argument("--secs", type=float, default=6.0)
p.add_argument("--envs", type=int, default=32)
a = p.parse_args()
torch.manual_seed(0)
cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = a.envs
m = cfg.commands["motion"]
m.motion_file = "docs/bwc_motion_v2.npz"; m.sampling_mode = "uniform"
if hasattr(m, "standing_start_prob"): m.standing_start_prob = 0.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"), clip_actions=load_rl_cfg(a.task).clip_actions)
runner = load_runner_cls(a.task)(env, asdict(load_rl_cfg(a.task)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / a.run / a.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
cmd = env.unwrapped.command_manager.get_term("motion")
robot = env.unwrapped.scene["robot"]
feet = [list(robot.body_names).index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
obs, _ = env.reset()
steps = int(a.secs / env.unwrapped.step_dt)
r0 = cmd.robot_anchor_pos_w[:, :2].clone(); f0 = cmd.anchor_pos_w[:, :2].clone()
alive = torch.ones(a.envs, dtype=torch.bool); life = torch.full((a.envs,), float(steps))
rd = torch.zeros(a.envs); fd = torch.zeros(a.envs); zmin = torch.full((a.envs,), 9.); zmax = torch.zeros(a.envs)
zref_min = torch.full((a.envs,), 9.); zref_max = torch.zeros(a.envs)
with torch.no_grad():
  for i in range(steps):
    rp = cmd.robot_anchor_pos_w[:, :2].clone(); fp = cmd.anchor_pos_w[:, :2].clone()
    obs, _, dones, _ = env.step(policy(obs))
    d = dones.bool()
    rd = torch.where(alive, (rp - r0).norm(dim=1), rd); fd = torch.where(alive, (fp - f0).norm(dim=1), fd)
    z = robot.data.body_link_pos_w[:, feet, 2]; zr = cmd.body_pos_w[:, [list(robot.body_names).index("L_ANKLE_P_LINK"), list(robot.body_names).index("R_ANKLE_P_LINK")], 2] if cmd.body_pos_w.shape[1] == len(robot.body_names) else None
    zmin = torch.where(alive, torch.minimum(zmin, z.min(1).values), zmin); zmax = torch.where(alive, torch.maximum(zmax, z.max(1).values), zmax)
    life[alive & d] = i; alive &= ~d
walk = fd > 0.15
print(f"\n{a.run} {a.checkpoint}  {a.secs:.0f} s depuis un etat de la reference, {int(walk.sum())}/{a.envs} en marche")
print(f"  deplacement reference  {fd[walk].mean():.2f} m   robot {rd[walk].mean():.2f} m   ratio {(rd[walk]/fd[walk]).mean():.2f}")
print(f"  survie {life[walk].mean()*env.unwrapped.step_dt:.1f} s   lever max-min pied {(zmax-zmin)[walk].mean()*100:.1f} cm")
