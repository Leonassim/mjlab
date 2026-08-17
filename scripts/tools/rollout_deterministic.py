"""Measure whether a policy really advances, deterministically.

Two traps this avoids. The CURRICULUM rewrites the command ranges at every
reset, so setting cfg.commands["twist"].ranges does nothing. And forcing the
command after env.step() does nothing either: command_manager.compute() runs
INSIDE step, just before the observation is built, so the only correct
injection point is _update_command itself.

Integrated displacement, not instantaneous velocity: velocity oscillates with
the step, position does not.

  uv run python scripts/tools/rollout_deterministic.py <checkpoint.pt> [speed]
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
DURATION_S = 5.0
POLICY_DT = 0.005


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  speeds = [float(s) for s in sys.argv[2:]] or [0.1, 0.2, 0.3]
  steps = int(DURATION_S / POLICY_DT)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 64
  # Vide, pas None : load_managers() fait len() dessus sans garde.
  cfg.curriculum = {}

  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")

  for fwd in speeds:
    # Injected in _update_command, the last thing to write vel_command_out before
    # the observation is built. Writing cmd.command after env.step() instead is a
    # no-op: command_manager.compute() runs inside step and overwrites it.
    target = torch.tensor([fwd, 0.0, 0.0], device=device)

    def forced_update(t: torch.Tensor = target) -> None:
      cmd.vel_command_b[:] = t
      cmd.vel_command_w[:] = t
      cmd.vel_command_out[:] = t

    cmd._update_command = forced_update  # type: ignore[method-assign]
    cmd.is_standing_env[:] = False

    env.reset()
    cmd.is_standing_env[:] = False
    obs = env.get_observations()
    if isinstance(obs, tuple):
      obs = obs[0]

    x0 = robot.data.root_link_pos_w[:, 0].clone()
    seen = []
    for _ in range(steps):
      seen.append(cmd.command[:, 0].clone())
      with torch.inference_mode():
        action = policy(obs)
      obs = env.step(action)[0]
      cmd.is_standing_env[:] = False

    seen_t = torch.stack(seen)
    travelled = (robot.data.root_link_pos_w[:, 0] - x0).cpu().numpy()
    expected = fwd * DURATION_S

    print(f"\ncommande {fwd} m/s pendant {DURATION_S} s -> attendu {expected:.2f} m")
    print(
      f"  commande vue par la policy : min {float(seen_t.min()):.3f} "
      f"max {float(seen_t.max()):.3f}   (doit valoir la consigne)"
    )
    for q in (10, 50, 90, 100):
      v = float(np.percentile(travelled, q))
      print(f"    p{q:<3d} {v:+.3f} m   {100 * v / expected:5.0f} %")
    print(
      f"  envs a plus de 50 % du suivi : "
      f"{int((travelled > 0.5 * expected).sum())}/{len(travelled)}"
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
