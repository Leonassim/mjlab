"""Does the robot advance inside the real training loop?

Measures displacement from within runner.learn rather than a hand-rolled loop:
same env construction as train.py, same wrapper, same transition collection.

Two metrics have already misled here, both for the same reason. sole_height_p50
is a median over all timesteps, with the feet on the ground most of the time.
error_vel_xy is an INSTANTANEOUS base velocity error, and a biped that walks
well oscillates every step.

  uv run python scripts/tools/displacement_in_training_loop.py <checkpoint.pt>
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
NUM_ENVS = 1024
ITERATIONS = 40

# Etage final atteint par le run 2026-08-12_20-36-28.
FINAL_STAGE = {
  "step": 0,
  "lin_vel_x": (-0.3, 0.3),
  "lin_vel_y": (-0.4, 0.4),
  "ang_vel_z": (-0.45, 0.45),
}


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = NUM_ENVS
  if cfg.curriculum and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [FINAL_STAGE]

  agent_cfg = load_rl_cfg(TASK)
  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)

  robot = env_raw.scene["robot"]
  cmd_term = env_raw.command_manager.get_term("twist")

  # Realised and demanded displacement per env. A reset teleports the robot, so
  # its increment must be dropped rather than counted as walked distance.
  prev_xy = robot.data.root_link_pos_w[:, :2].clone()
  moved = torch.zeros((NUM_ENVS, 2), device=device)
  want = torch.zeros((NUM_ENVS, 2), device=device)
  standing_seen = torch.zeros(NUM_ENVS, device=device)
  nsteps = 0

  raw_step = env.step

  def instrumented(actions):
    nonlocal prev_xy, nsteps
    out = raw_step(actions)
    dones = out[2].bool()
    xy = robot.data.root_link_pos_w[:, :2]
    d = xy - prev_xy
    # An env that just reset has jumped; ignore its increment.
    d[dones] = 0.0
    moved.add_(d)
    want.add_(cmd_term.command[:, :2] * env_raw.step_dt)
    standing_seen.add_(cmd_term.is_standing_env.float())
    prev_xy = xy.clone()
    nsteps += 1
    return out

  env.step = instrumented  # type: ignore[method-assign]

  with tempfile.TemporaryDirectory() as log_dir:
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    d = asdict(agent_cfg)
    d["logger"] = "tensorboard"
    runner = runner_cls(env, d, log_dir, device)
    runner.load(ckpt, map_location=device)
    runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=True)

  m = moved.cpu().numpy()
  w = want.cpu().numpy()
  standing_frac = (standing_seen / max(nsteps, 1)).cpu().numpy()

  demand = np.linalg.norm(w, axis=1)
  walking = (standing_frac < 0.2) & (demand > 0.5)
  print(f"\n{nsteps} pas collectes, {NUM_ENVS} envs")
  print(f"{int(walking.sum())}/{NUM_ENVS} envs avec une vraie demande de marche")
  if walking.sum() == 0:
    print("aucune demande de marche : rien a conclure")
    return 0

  unit = w[walking] / demand[walking][:, None]
  along = (m[walking] * unit).sum(axis=1)
  ratio = along / demand[walking]
  print(f"demande moyenne {demand[walking].mean():.2f} m")
  print(f"{'percentile':>12s} {'realise':>10s} {'suivi':>8s}")
  for q in (10, 50, 90):
    print(
      f"{'p' + str(q):>12s} {float(np.percentile(along, q)):+9.3f}m "
      f"{100 * float(np.percentile(ratio, q)):7.0f} %"
    )
  print(f"envs au-dessus de 50 % : {int((ratio > 0.5).sum())}/{int(walking.sum())}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
