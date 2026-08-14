"""Suivi de vitesse avec les commandes NATURELLES de l'env d'entrainement.

Complement de why_video_walks.py. Ici on ne force rien du tout : curriculum
actif, commandes echantillonnees, bruit d'exploration -- exactement la
configuration que VideoRecorder filme pour wandb. On regarde, environnement par
environnement, le deplacement realise contre le deplacement demande.

C'est le seul moyen de distinguer deux explications d'un robot qui a l'air de
marcher a l'ecran mais mesure 2 % de suivi en commande forcee :

  - la policy suit bien sa commande, et c'est le protocole en commande
    constante forcee qui est pathologique
  - la policy pietine, et la vidéo montre un pas sur place

  uv run python scripts/tools/natural_command_tracking.py <checkpoint.pt>
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
NUM_ENVS = 256


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = NUM_ENVS
  cfg.episode_length_s = max(cfg.episode_length_s, 2 * DURATION_S)

  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")

  env.reset()
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  x0 = robot.data.root_link_pos_w[:, :2].clone()
  cmd_sum = torch.zeros((NUM_ENVS, 2), device=device)
  steps = int(DURATION_S / POLICY_DT)
  for _ in range(steps):
    # Integrer la commande vue, elle change en cours de route (resampling).
    cmd_sum += cmd.command[:, :2] * POLICY_DT
    with torch.inference_mode():
      action = policy(obs, stochastic_output=True)
    obs = env.step(action)[0]

  moved = (robot.data.root_link_pos_w[:, :2] - x0).cpu().numpy()
  want = cmd_sum.cpu().numpy()
  standing = cmd.is_standing_env.cpu().numpy()

  demand = np.linalg.norm(want, axis=1)
  walking = (~standing) & (demand > 0.25)  # au moins 25 cm demandes sur 5 s
  print(f"{int(standing.sum())}/{NUM_ENVS} envs en consigne immobile")
  print(f"{int(walking.sum())}/{NUM_ENVS} envs avec une vraie demande de marche")

  if walking.sum() == 0:
    print("aucun env ne recoit de demande de marche significative")
    return 0

  # Projection du deplacement realise sur la direction demandee : c'est le
  # chiffre honnete, un robot qui derive de cote ne doit pas etre credite.
  unit = want[walking] / demand[walking][:, None]
  along = (moved[walking] * unit).sum(axis=1)
  ratio = along / demand[walking]

  print(f"\ndemande moyenne {demand[walking].mean():.2f} m sur {DURATION_S} s")
  print(f"{'percentile':>12s} {'realise':>10s} {'suivi':>8s}")
  for q in (10, 50, 90, 100):
    print(
      f"{'p' + str(q):>12s} {float(np.percentile(along, q)):+9.3f}m "
      f"{100 * float(np.percentile(ratio, q)):7.0f} %"
    )
  print(f"\nenvs au-dessus de 50 % du suivi : {int((ratio > 0.5).sum())}/{int(walking.sum())}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
