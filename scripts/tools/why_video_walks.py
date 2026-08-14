"""Why the wandb video walks and play does not.

wandb videos do NOT come from play: train.py wraps the TRAINING env in
VideoRecorder and films env 0. Three differences, not one -- sampled actions,
observation corruption, recovery pushes.

Forcing the command after env.step() forces nothing: command_manager.compute()
runs inside step, just before the observation is built, so the injection point
is _update_command. Re-reading cmd.command right after writing it is a
tautological check.

  uv run python scripts/tools/why_video_walks.py <checkpoint.pt> [speed]
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
NUM_ENVS = 64


def run(
  ckpt: str, fwd: float, play: bool, stochastic: bool, video: str | None = None
) -> np.ndarray:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=play)
  cfg.scene.num_envs = NUM_ENVS
  # Empty, not None: load_managers() calls len() on it without a guard.
  cfg.curriculum = {}
  # Otherwise the training env cuts the episode short.
  cfg.episode_length_s = max(cfg.episode_length_s, 2 * DURATION_S)

  env_raw = ManagerBasedRlEnv(
    cfg, device=device, render_mode="rgb_array" if video else None
  )
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")

  # Forcing the command AFTER env.step() does nothing: command_manager.compute()
  # runs INSIDE step, just before observation_manager.compute(), so it rewrites
  # the command before the observation is built. _update_command is the only
  # correct injection point.
  target = torch.tensor([fwd, 0.0, 0.0], device=device)

  def forced_update() -> None:
    cmd.vel_command_b[:] = target
    cmd.vel_command_w[:] = target
    cmd.vel_command_out[:] = target

  cmd._update_command = forced_update  # type: ignore[method-assign]
  # The standing mask would zero 40% of the envs; clear it so nothing else in
  # the env believes a stop was requested.
  cmd.is_standing_env[:] = False

  env.reset()
  cmd.is_standing_env[:] = False
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  x0 = robot.data.root_link_pos_w[:, 0].clone()
  # Read at the top of the loop, hence after the previous step's compute: this
  # is the value the env produced, not the one we just wrote ourselves.
  seen_min, seen_max = 1e9, -1e9
  frames: list[np.ndarray] = []
  for i in range(int(DURATION_S / POLICY_DT)):
    seen_min = min(seen_min, float(cmd.command[:, 0].min()))
    seen_max = max(seen_max, float(cmd.command[:, 0].max()))
    with torch.inference_mode():
      action = policy(obs, stochastic_output=True) if stochastic else policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    # Every 4th frame -> 50 fps for a 200 Hz policy step.
    if video and i % 4 == 0:
      frames.append(env_raw.render())

  if video:
    import mediapy as media

    media.write_video(video, frames, fps=50)
    print(f"  video ecrite : {video}")

  travelled = (robot.data.root_link_pos_w[:, 0] - x0).cpu().numpy()
  assert abs(seen_min - fwd) < 1e-6 and abs(seen_max - fwd) < 1e-6, (
    f"commande vue {seen_min}..{seen_max} au lieu de {fwd}"
  )
  env_raw.close()
  return travelled


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  expected = fwd * DURATION_S

  print(f"commande {fwd} m/s pendant {DURATION_S} s -> attendu {expected:.2f} m")
  print(f"{'condition':38s} {'p50':>9s} {'p90':>9s} {'suivi p50':>10s}")
  for play in (True, False):
    for stochastic in (False, True):
      label = (
        f"{'play=True (deploiement)' if play else 'play=False (=video wandb)'}"
        f" + {'stochastique' if stochastic else 'deterministe'}"
      )
      tag = f"{'play' if play else 'train'}_{'stoch' if stochastic else 'det'}"
      d = run(ckpt, fwd, play, stochastic, video=f"/tmp/rhps1_{tag}.mp4")
      p50, p90 = float(np.percentile(d, 50)), float(np.percentile(d, 90))
      print(f"{label:38s} {p50:+8.3f}m {p90:+8.3f}m {100 * p50 / expected:9.0f} %")
  return 0


if __name__ == "__main__":
  sys.exit(main())
