"""Replay a checkpoint in front of a FIXED camera.

Training videos use viewer.origin_type = ASSET_BODY, so the robot stays centred
whether it advances or marches in place. With a WORLD camera the question is
binary: it leaves the frame or it does not. Renders the policy and a dragged
control for comparison.

  uv run python scripts/tools/render_fixed_camera.py <checkpoint.pt> [speed]
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

TASK = "Mjlab-Velocity-Flat-RHPS1"
DURATION_S = 6.0
POLICY_DT = 0.005


def run(ckpt: str, fwd: float, drag: bool, out: str) -> float:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = 1
  cfg.curriculum = {}
  cfg.episode_length_s = 4 * DURATION_S
  if drag:
    cfg.terminations = {}

  # Fixed side camera framing the start: 1 m of travel visibly crosses it.
  cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  cfg.viewer.lookat = (0.8, 0.0, 0.6)
  cfg.viewer.distance = 4.5
  cfg.viewer.elevation = -12.0
  cfg.viewer.azimuth = 90.0

  env_raw = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")
  target = torch.tensor([fwd, 0.0, 0.0], device=device)

  def forced_update() -> None:
    cmd.vel_command_b[:] = target
    cmd.vel_command_w[:] = target
    cmd.vel_command_out[:] = target

  cmd._update_command = forced_update  # type: ignore[method-assign]
  cmd.is_standing_env[:] = False

  env.reset()
  cmd.is_standing_env[:] = False
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  vel = torch.zeros((1, 6), device=device)
  vel[:, 0] = fwd
  x0 = float(robot.data.root_link_pos_w[0, 0])

  frames = []
  for i in range(int(DURATION_S / POLICY_DT)):
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    if drag:
      robot.write_root_link_velocity_to_sim(vel)
    if i % 4 == 0:
      frames.append(env_raw.render())

  import mediapy as media

  media.write_video(out, frames, fps=50)
  travelled = float(robot.data.root_link_pos_w[0, 0]) - x0
  print(f"{out} : deplacement reel {travelled:+.2f} m en {DURATION_S} s")
  env_raw.close()
  return travelled


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  print(f"commande {fwd} m/s, camera FIXE, attendu {fwd * DURATION_S:.1f} m\n")
  run(ckpt, fwd, drag=False, out="/tmp/rhps1_fixe_policy.mp4")
  run(ckpt, fwd, drag=True, out="/tmp/rhps1_fixe_temoin.mp4")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
