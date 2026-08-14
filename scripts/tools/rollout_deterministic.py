"""Mesurer si une policy avance vraiment, en deterministe.

Pourquoi ce script existe. Les videos d'entrainement contiennent le bruit
d'exploration : les jambes bougent, ca ressemble a une marche, et l'ONNX
deploye -- lui deterministe -- peut pietiner sur place. Le 2026-08-14 c'est
exactement ce qui s'est passe : le run 2026-08-12_20-36-28 avait l'air de
marcher a l'ecran et suivait 2 % de sa commande en deterministe.

Deux pieges que ce script evite, tous deux rencontres en essayant de faire la
mesure a la main :

1. Le CURRICULUM reecrit les plages de commande a chaque reset. Regler
   cfg.commands["twist"].ranges ne sert donc a rien : on croit mesurer 0.2 m/s
   et la commande vue par le reseau va de -0.26 a +0.30. Il faut vider le
   curriculum (cfg.curriculum = {}) ET forcer la commande a chaque pas.
2. rel_standing_envs met par defaut 40 % des environnements en consigne
   "immobile". Sans les neutraliser, la moyenne est ecrasee par des envs a qui
   on n'a jamais demande d'avancer.

Le script verifie et affiche la commande reellement vue par la policy, pour que
le lecteur puisse constater qu'elle vaut bien la consigne et pas autre chose.

On mesure le DEPLACEMENT integre, pas la vitesse instantanee : la vitesse
oscille avec le pas et sa moyenne est bruitee, la position ne ment pas.

  uv run python scripts/tools/rollout_deterministic.py <checkpoint.pt> [vitesse]
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
    env.reset()

    def force() -> None:
      cmd.command[:, 0] = fwd
      cmd.command[:, 1] = 0.0
      cmd.command[:, 2] = 0.0

    force()
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
      force()

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
