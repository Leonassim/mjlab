"""Comparer les observations d'un rollout a celles vues pendant l'entrainement.

Le checkpoint transporte les statistiques figees d'EmpiricalNormalization :
c'est un enregistrement direct de la distribution d'observations sur laquelle la
policy a ete entrainee (ici count = 2.9e9 echantillons). On peut donc rejouer un
rollout, mesurer la distribution qu'il produit, et exprimer l'ecart en z-score
par dimension :

    z = (moyenne du rollout - _mean du checkpoint) / _std du checkpoint

Une dimension a |z| de l'ordre de 1 est normale. Une dimension a |z| de 10 dit
que la policy recoit, sur cette entree precise, quelque chose qu'elle n'a jamais
vu -- et le nom du terme d'observation designe le coupable.

C'est le seul diagnostic qui compare le banc de mesure a l'ENTRAINEMENT REEL et
non a une autre execution du banc.

  uv run python scripts/tools/obs_drift.py <checkpoint.pt> [--train]
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


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  play = "--train" not in sys.argv

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=play)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}
  cfg.episode_length_s = max(cfg.episode_length_s, 2 * DURATION_S)

  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  norm = policy.obs_normalizer
  ref_mean = norm._mean.squeeze(0).float().cpu().numpy()
  ref_std = norm._std.squeeze(0).float().cpu().numpy()
  print(f"normalisation restauree : count={int(norm.count)}")
  if int(norm.count) == 0:
    print("  ATTENTION : count nul, la normalisation n'a PAS ete rechargee")

  # Noms de chaque dimension, pour pouvoir nommer le coupable.
  om = env_raw.observation_manager
  group = "actor"
  terms: list[tuple[str, int]] = []
  names: list[str] = []
  for g in ("actor",):
    for term, dim in zip(
      om.active_terms.get(g, []), om.group_obs_term_dim.get(g, []), strict=True
    ):
      n = int(np.prod(dim))
      terms.append((f"{g}/{term}", n))
      names.extend(f"{g}/{term}[{i}]" for i in range(n))

  cmd = env_raw.command_manager.get_term("twist")
  target = torch.tensor([0.2, 0.0, 0.0], device=device)

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

  # Exactement ce que le reseau normalise : la concatenation de ses obs_groups,
  # dans l'ordre du modele (cf. MLPModel.get_latent).
  groups = list(policy.obs_groups)
  print(f"groupes concatenes par le modele : {groups}")

  acc = []
  for _ in range(int(DURATION_S / POLICY_DT)):
    latent = torch.cat([obs[g] for g in groups], dim=-1)
    acc.append(latent.float().mean(dim=0).cpu().numpy())
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False

  # Les 200 premiers pas contiennent le transitoire de reset.
  got = np.stack(acc)[200:].mean(axis=0)

  if len(names) != got.shape[0]:
    print(f"  (noms {len(names)} vs dims {got.shape[0]} -- indices bruts)")
    names = [f"dim[{i}]" for i in range(got.shape[0])]

  z = (got - ref_mean) / (ref_std + 1e-2)
  order = np.argsort(-np.abs(z))

  print(f"\nconfig : {'play' if play else 'entrainement'}")
  print(f"|z| median {np.median(np.abs(z)):.2f}   |z| max {np.abs(z).max():.1f}")
  print(f"dimensions a |z| > 3 : {int((np.abs(z) > 3).sum())} / {len(z)}\n")
  print(f"{'dimension':32s} {'rollout':>10s} {'train':>10s} {'std':>9s} {'z':>8s}")
  for i in order[:25]:
    print(
      f"{names[i]:32s} {got[i]:+10.3f} {ref_mean[i]:+10.3f} "
      f"{ref_std[i]:9.3f} {z[i]:+8.1f}"
    )

  # Vue agregee par terme : plus lisible qu'une liste de dimensions.
  print(f"\n{'terme':32s} {'|z| max':>9s} {'|z| moyen':>10s}")
  start = 0
  for term, n in terms:
    if start + n > len(z):
      break
    zz = np.abs(z[start : start + n])
    print(f"{term:32s} {zz.max():9.1f} {zz.mean():10.2f}")
    start += n
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
