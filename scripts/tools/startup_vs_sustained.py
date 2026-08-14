"""La policy sait-elle DEMARRER une marche, ou seulement l'entretenir ?

Les vidéos d'entrainement demarrent a un instant quelconque de l'episode : env 0
est deja en mouvement depuis plusieurs secondes quand l'enregistrement commence.
`uv run play`, mc_mujoco et tout banc de mesure demarrent au contraire depuis le
reset, robot immobile. Si la policy a appris a entretenir une allure sans savoir
l'amorcer, les deux observations -- "ca marche dans la vidéo" et "ca se bloque
au lancement" -- sont vraies en meme temps.

Trois conditions, meme checkpoint, meme commande, config d'entrainement :

  depart arrete, 5 s     le protocole utilise jusqu'ici
  depart arrete, 20 s    laisse le temps d'amorcer (episode d'entrainement)
  depart lance, 5 s      vitesse initiale = commande, l'etat des vidéos

On mesure le deplacement sur les 5 DERNIERES secondes de chaque condition :
c'est le regime etabli, pas le transitoire.

  uv run python scripts/tools/startup_vs_sustained.py <checkpoint.pt> [vitesse]
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
POLICY_DT = 0.005
NUM_ENVS = 64
TAIL_S = 5.0


def run(
  ckpt: str, fwd: float, duration_s: float, kick: bool, video: str | None
) -> tuple[np.ndarray, np.ndarray]:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}
  cfg.episode_length_s = max(cfg.episode_length_s, 2 * duration_s)

  env_raw = ManagerBasedRlEnv(
    cfg, device=device, render_mode="rgb_array" if video else None
  )
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

  if kick:
    # Vitesse initiale egale a la commande : l'etat dans lequel se trouve env 0
    # quand VideoRecorder commence a filmer, au lieu d'un robot a l'arret.
    vel = torch.zeros((NUM_ENVS, 6), device=device)
    vel[:, 0] = fwd
    robot.write_root_link_velocity_to_sim(vel)

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  steps = int(duration_s / POLICY_DT)
  tail_start = steps - int(TAIL_S / POLICY_DT)
  x_tail = None
  frames: list[np.ndarray] = []
  for i in range(steps):
    if i == tail_start:
      x_tail = robot.data.root_link_pos_w[:, 0].clone()
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    if video and i % 4 == 0:
      frames.append(env_raw.render())

  assert x_tail is not None
  tail = (robot.data.root_link_pos_w[:, 0] - x_tail).cpu().numpy()
  vel_now = robot.data.root_link_lin_vel_b[:, 0].cpu().numpy()

  if video:
    import mediapy as media

    media.write_video(video, frames, fps=50)
    print(f"    video : {video}")

  env_raw.close()
  return tail, vel_now


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  expected = fwd * TAIL_S

  conditions = [
    ("depart arrete, 5 s", 5.0, False, "/tmp/rhps1_start_5s.mp4"),
    ("depart arrete, 20 s", 20.0, False, "/tmp/rhps1_start_20s.mp4"),
    ("depart lance, 5 s", 5.0, True, "/tmp/rhps1_kick_5s.mp4"),
  ]

  print(
    f"commande {fwd} m/s ; deplacement mesure sur les {TAIL_S} dernieres "
    f"secondes -> attendu {expected:.2f} m\n"
  )
  print(f"{'condition':24s} {'p50':>9s} {'p90':>9s} {'suivi p50':>10s} {'v_x fin':>9s}")
  for label, dur, kick, video in conditions:
    tail, vel = run(ckpt, fwd, dur, kick, video)
    p50, p90 = float(np.percentile(tail, 50)), float(np.percentile(tail, 90))
    print(
      f"{label:24s} {p50:+8.3f}m {p90:+8.3f}m {100 * p50 / expected:9.0f} % "
      f"{float(np.median(vel)):+8.3f}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
