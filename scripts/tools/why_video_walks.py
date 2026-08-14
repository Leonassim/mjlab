"""Pourquoi la video wandb marche et 'play' ne marche pas.

Les videos wandb ne viennent PAS de play : train.py enveloppe l'env
d'ENTRAINEMENT (play=False) dans VideoRecorder et filme l'environnement 0. Trois
differences avec play, pas une seule :

  1. action ECHANTILLONNEE (bruit d'exploration) vs deterministe
  2. corruption des observations active vs desactivee
  3. poussees de recuperation actives vs desactivees

Ce script separe les deux premieres, qui sont les seules a pouvoir faire
avancer un robot qui piétine. On mesure le deplacement integre sur 5 s, commande
forcee et verifiee, curriculum vide (sinon il reecrit les plages a chaque reset).

  uv run python scripts/tools/why_video_walks.py <checkpoint.pt> [vitesse]
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
  # Vide, pas None : load_managers() fait len() dessus sans garde.
  cfg.curriculum = {}
  # Sinon l'env d'entrainement coupe l'episode avant les 5 s.
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

  # Forcer la commande APRES env.step() ne sert a rien : command_manager.compute()
  # tourne A L'INTERIEUR de step, juste avant observation_manager.compute(), donc
  # il reecrit la commande avant que l'observation soit construite. Le seul point
  # d'injection correct est _update_command lui-meme, qui est la derniere chose a
  # ecrire vel_command_out avant l'observation.
  target = torch.tensor([fwd, 0.0, 0.0], device=device)

  def forced_update() -> None:
    cmd.vel_command_b[:] = target
    cmd.vel_command_w[:] = target
    cmd.vel_command_out[:] = target

  cmd._update_command = forced_update  # type: ignore[method-assign]
  # La consigne "immobile" mettrait 40 % des envs a zero ; on la neutralise en
  # plus, pour que rien d'autre dans l'env (metriques, recompenses) ne croie
  # qu'on demande l'arret.
  cmd.is_standing_env[:] = False

  env.reset()
  cmd.is_standing_env[:] = False
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  x0 = robot.data.root_link_pos_w[:, 0].clone()
  # Lu en debut de boucle, donc apres le compute() du step precedent : c'est bien
  # la valeur produite par l'env, pas celle qu'on vient d'ecrire soi-meme.
  seen_min, seen_max = 1e9, -1e9
  frames: list[np.ndarray] = []
  for i in range(int(DURATION_S / POLICY_DT)):
    seen_min = min(seen_min, float(cmd.command[:, 0].min()))
    seen_max = max(seen_max, float(cmd.command[:, 0].max()))
    with torch.inference_mode():
      action = policy(obs, stochastic_output=True) if stochastic else policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    # 1 image sur 4 -> 50 im/s pour un pas de politique a 200 Hz.
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
