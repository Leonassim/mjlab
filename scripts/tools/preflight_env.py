"""Verifier qu'un robot inerte tient debout avant de lancer un entrainement.

Pourquoi ce script existe. Le 2026-08-12, le run 2026-08-12_16-04-55 a tourne
une heure pour rien : `link_inertia` avait recu alpha_range=(0.9, 1.1) alors
que pseudo_inertia prend alpha en logarithme, donc masse et inertie etaient
multipliees par exp(2*alpha) = 6 a 9. Le robot pesait 350 a 520 kg et
s'effondrait en 1.4 s quoi que fasse la politique. Rien dans les courbes ne
criait au probleme : la recompense montait (-84 -> -46), simplement parce que
des episodes plus courts accumulent moins de penalites. Seule la longueur
d'episode, figee a 70 pas de l'iteration 50 a 492, disait la verite.

Le test tient en une phrase : un robot debout a qui on ne demande rien ne doit
pas tomber. Une action nulle veut dire "cible = posture par defaut", donc le
robot doit rester en equilibre indefiniment. S'il tombe, la faute est dans
l'environnement et aucune politique ne la rattrapera.

Ca ne remplace pas de lire la doc des fonctions de randomisation en amont --
c'est ce que j'aurais du faire -- mais ca attrape la classe entiere des
erreurs d'unite et d'echelle, qui sont muettes autrement.

Usage :
    uv run python scripts/tools/preflight_env.py Mjlab-Velocity-Flat-RHPS1
    uv run python scripts/tools/preflight_env.py <tache> --ablate   # bissection

Sort 0 si le robot survit, 1 sinon. Avec --ablate, retire chaque evenement a
tour de role et signale ceux dont l'absence restaure la stabilite : c'est ce
qui a isole link_inertia en huit essais.
"""

from __future__ import annotations

import argparse
import sys

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def survival(task: str, drop_event: str | None, num_envs: int, steps: int):
  """Renvoie (survie moyenne en pas, nombre d'environnements encore debout)."""
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  if drop_event is not None:
    if cfg.events is None or drop_event not in cfg.events:
      raise KeyError(f"evenement absent de la config : {drop_event}")
    cfg.events.pop(drop_event)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    alive = torch.ones(env.num_envs, dtype=torch.bool)
    life = torch.zeros(env.num_envs)
    for k in range(steps):
      _, _, terminated, truncated, _ = env.step(action)
      life[alive] = k + 1
      alive &= ~(terminated | truncated)
      if not alive.any():
        break
    return float(life.mean()), int(alive.sum())
  finally:
    del env


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("task")
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=300)
  p.add_argument(
    "--ablate",
    action="store_true",
    help="retirer chaque evenement a tour de role pour isoler le coupable",
  )
  args = p.parse_args()

  mean, standing = survival(args.task, None, args.num_envs, args.steps)
  print(f"config complete : survie {mean:.1f}/{args.steps} pas, "
        f"{standing}/{args.num_envs} encore debout")

  # Deliberately loose: this looks for an env that kills the robot, not one that
  # makes it wobble. A guard that cries wolf gets ignored.
  ok = standing >= 0.9 * args.num_envs
  if ok:
    print("OK : un robot inerte tient debout, l'environnement est lancable.")
    return 0

  print("ECHEC : le robot tombe sans qu'on lui demande rien. "
        "Aucune politique ne rattrapera ca -- ne pas lancer l'entrainement.")

  if args.ablate:
    cfg = load_env_cfg(args.task)
    names = list(cfg.events) if cfg.events else []
    print(f"\nbissection sur {len(names)} evenements :")
    for name in names:
      try:
        m, s = survival(args.task, name, args.num_envs, args.steps)
      except Exception as exc:  # un evenement non retirable n'arrete pas la passe
        print(f"  sans {name:22s} non testable ({type(exc).__name__})")
        continue
      verdict = "  <-- COUPABLE" if s >= 0.9 * args.num_envs else ""
      print(f"  sans {name:22s} survie {m:6.1f}  debout {s:3d}/{args.num_envs}{verdict}")
  else:
    print("Relancer avec --ablate pour isoler l'evenement responsable.")
  return 1


if __name__ == "__main__":
  sys.exit(main())
