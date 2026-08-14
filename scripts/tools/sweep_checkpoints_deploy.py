"""Compare checkpoints of one run under DEPLOYMENT conditions.

play=True, deterministic policy, command forced through _update_command.
Body-frame velocity and path length, not world x, which yaw falsifies.

  uv run python scripts/tools/sweep_checkpoints_deploy.py <run_dir> <it> [<it> ...]
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
DURATION_S = 6.0
POLICY_DT = 0.005
NUM_ENVS = 64
WARMUP = 200
FWD = 0.2


def main() -> int:
  if len(sys.argv) < 3:
    raise SystemExit(__doc__)
  run_dir = Path(sys.argv[1])
  iters = sys.argv[2:]

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}

  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")
  target = torch.tensor([FWD, 0.0, 0.0], device=device)

  def forced_update() -> None:
    cmd.vel_command_b[:] = target
    cmd.vel_command_w[:] = target
    cmd.vel_command_out[:] = target

  cmd._update_command = forced_update  # type: ignore[method-assign]

  window = DURATION_S - WARMUP * POLICY_DT
  print(f"commande {FWD} m/s, fenetre utile {window:.1f} s")
  print(f"chemin attendu si le robot suit : {FWD * window:.2f} m\n")
  print(f"{'checkpoint':>12s} {'v_x p50':>9s} {'v_x p90':>9s} {'chemin p50':>11s}")

  for it in iters:
    ckpt = run_dir / f"model_{it}.pt"
    if not ckpt.exists():
      print(f"{it:>12s}  absent")
      continue
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd.is_standing_env[:] = False
    env.reset()
    cmd.is_standing_env[:] = False
    obs = env.get_observations()
    if isinstance(obs, tuple):
      obs = obs[0]

    prev = robot.data.root_link_pos_w[:, :2].clone()
    path = torch.zeros(NUM_ENVS, device=device)
    vx = []
    for i in range(int(DURATION_S / POLICY_DT)):
      with torch.inference_mode():
        action = policy(obs)
      obs = env.step(action)[0]
      cmd.is_standing_env[:] = False
      xy = robot.data.root_link_pos_w[:, :2]
      if i >= WARMUP:
        path += torch.norm(xy - prev, dim=-1)
        vx.append(robot.data.root_link_lin_vel_b[:, 0].clone())
      prev = xy.clone()

    v = torch.stack(vx).mean(dim=0).cpu().numpy()
    p = path.cpu().numpy()
    print(
      f"{it:>12s} {float(np.percentile(v, 50)):+8.3f} "
      f"{float(np.percentile(v, 90)):+8.3f} {float(np.percentile(p, 50)):10.3f}m",
      flush=True,
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
