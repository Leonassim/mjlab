"""Control for video_ground_motion.py: a robot that definitely advances.

Measuring no ground scroll proves nothing unless the tool can measure scroll at
all. Same scene, same tracking camera, but the robot is dragged forward at a
constant speed, so the ground MUST scroll.

  uv run python scripts/tools/render_control_translation.py <checkpoint.pt> [speed]
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
DURATION_S = 3.0
POLICY_DT = 0.005
OUT = "/tmp/rhps1_control_drag.mp4"


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = 16
  cfg.episode_length_s = 10 * DURATION_S
  cfg.terminations = {}  # le trainage n'est pas physique, ne pas terminer dessus

  env_raw = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)
  robot = env_raw.scene["robot"]

  env.reset()
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  vel = torch.zeros((cfg.scene.num_envs, 6), device=device)
  vel[:, 0] = fwd

  frames = []
  for i in range(int(DURATION_S / POLICY_DT)):
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    # Force the forward velocity after each step: the robot crosses the scene
    # whatever the policy does.
    robot.write_root_link_velocity_to_sim(vel)
    if i % 4 == 0:
      frames.append(env_raw.render())

  import mediapy as media

  media.write_video(OUT, frames, fps=50)
  x = float(robot.data.root_link_pos_w[0, 0])
  print(f"temoin ecrit : {OUT}")
  print(f"deplacement reel de l'env 0 : {x:+.2f} m en {DURATION_S} s")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
