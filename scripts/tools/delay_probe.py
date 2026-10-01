"""Standing at command 0 under an added actuator delay.

Reproduces the real-robot oscillation of 2026-09-30: settle 3 s with actions held,
then the policy stands at command 0 for 6 s. Prints the RMS target change per
policy step in 1 s windows, per lag.

  uv run python scripts/tools/delay_probe.py <run> <model.pt> [lag_ms ...]
"""
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
lags_ms = [float(x) for x in sys.argv[3:]] or [0, 10, 20, 30, 45, 60]
N = 16
torch.manual_seed(0)

for lag_ms in lags_ms:
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = N
  for k in list(cfg.terminations.keys()):
    if k != "time_out":
      cfg.terminations.pop(k)
  cfg.commands["motion"].motion_file = "docs/bwc_motion_v2.npz"
  cfg.observations["actor"].terms["velocity_command"].func = (
    lambda env, command_name: torch.zeros(env.num_envs, 3, device=env.device)
  )
  lag = int(round(lag_ms / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
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
  # Envs start apart (+-0.03 rad, +-0.1 m/s), otherwise they are one sample repeated.
  q0 = q0 + (torch.rand_like(q0) * 2 - 1) * 0.03
  robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
  root = robot.data.default_root_state.clone()
  root[:, :3] += u.scene.env_origins
  root[:, 7:9] = (torch.rand(N, 2) * 2 - 1) * 0.1
  robot.write_root_state_to_sim(root)
  with torch.no_grad():
    for _ in range(int(3.0 / dt)):
      obs, _, _, _ = env.step(torch.zeros(N, env.num_actions))
    T, Z = [], []
    for _ in range(int(6.0 / dt)):
      obs, _, _, _ = env.step(pol(obs))
      T.append(robot.data.joint_pos_target.numpy().copy())
      Z.append(robot.data.root_link_pos_w[:, 2].numpy().copy())
  T, Z = np.stack(T), np.stack(Z)
  n = int(1.0 / dt)
  up = Z.min(0) >= 0.6
  rms = [np.degrees(np.sqrt((np.diff(T[i * n:(i + 1) * n][:, up], axis=0) ** 2).mean())) if up.any() else float('nan') for i in range(6)]
  fell = int((Z.min(0) < 0.6).sum())
  print(f"lag {lag_ms:4.0f} ms ({lag:2d} steps): rms dq_target/step per second "
        + " ".join(f"{r:6.3f}" for r in rms) + f"  | fell {fell}/{N} (rms over standing envs)", flush=True)
  env.close()
