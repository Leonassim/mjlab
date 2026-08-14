"""Temoin pour video_ground_motion.py : un robot qui avance a coup sur.

Mesurer un defilement de sol nul ne prouve rien si l'outil est incapable de
mesurer un defilement tout court -- le sol du decor a un contraste faible. Ce
script produit donc la reference manquante : la meme scene, la meme camera
suiveuse, mais le robot est TRAINE en avant a vitesse constante en ecrivant sa
vitesse de base a chaque pas. Le sol DOIT defiler.

Si l'outil voit ce defilement-la et pas celui de la video d'entrainement, le
verdict sur la video est valide. S'il ne voit ni l'un ni l'autre, l'outil est
aveugle et il faut le jeter.

  uv run python scripts/tools/render_control_translation.py <checkpoint.pt> [vitesse]
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
    # Imposer la vitesse d'avance apres chaque pas : le robot traverse le decor
    # quoi que fasse la politique.
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
