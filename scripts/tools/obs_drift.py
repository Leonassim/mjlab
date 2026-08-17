"""Compare a rollout's observations against the training distribution.

The checkpoint carries EmpiricalNormalization's frozen statistics, which are a
direct record of the distribution the policy trained on. Express the gap as a
per-dimension z score and the offending observation term names itself.

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

  # Per-dimension names, so the offending term can be named.
  om = env_raw.observation_manager
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

  # Exactly what the network normalises: its obs_groups concatenated in model
  # order, see MLPModel.get_latent.
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

  # The first 200 steps hold the reset transient.
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
