"""Quelle excursion articulaire la politique COMMANDE-t-elle reellement ?

Le lever de pied ne bouge pas malgre le budget de recompense. Avant de retirer
un quatrieme terme, verifier que le reseau peut seulement commander le
mouvement : q_cible = action * echelle + q0, donc l'excursion accessible est
bornee par l'amplitude des actions que le reseau emet.

  uv run python scripts/tools/action_reach.py <run> <model.pt> [--cmd 0.2]
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--cmd", type=float, default=0.2)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--envs", type=int, default=16)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  a = p.parse_args()
  device = "cpu"

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = a.envs
  for k in ("standing_envs", "command_vel"):
    getattr(cfg, "curriculum", {}).pop(k, None)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device=device),
    clip_actions=load_rl_cfg(TASK).clip_actions,
  )
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(load_rl_cfg(TASK)), device=device
  )
  runner.load(str(Path(a.root) / a.run / a.checkpoint),
              load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  term = env.unwrapped.command_manager.get_term("twist")
  v = torch.tensor([a.cmd, 0.0, 0.0], device=device)

  def _fixed(ids):
    term.vel_command_b[ids] = v
    term.is_standing_env[ids] = False
    term.is_heading_env[ids] = False

  term._resample_command = _fixed
  act_term = env.unwrapped.action_manager.get_term("joint_pos")
  robot = env.unwrapped.scene["robot"]
  names = list(robot.joint_names)
  order = list(getattr(act_term, "_joint_names", names))
  scale = act_term._scale.clone().cpu().numpy().reshape(-1)
  q0 = robot.data.default_joint_pos[0].cpu().numpy()

  obs, _ = env.reset()
  _fixed(slice(None))
  acts, qs = [], []
  with torch.no_grad():
    for i in range(a.steps):
      act = policy(obs)
      obs, _, _, _ = env.step(act)
      if i > a.steps // 4:
        acts.append(act.cpu().numpy())
        qs.append(robot.data.joint_pos.cpu().numpy())
  A = np.concatenate(acts, 0)
  Q = np.concatenate(qs, 0)

  print(f"\n{a.run} {a.checkpoint}  commande {a.cmd} m/s  {A.shape[0]} echantillons")
  print(f"\n{'joint':14} {'|a| p99':>8} {'echelle':>9} {'excursion cmd':>14} "
        f"{'excursion mesuree':>18} {'BWC':>7}")
  BWC = {"CROTCH_P": 19.5, "KNEE_P": 16.1, "ANKLE_P": 12.0}
  for j, n in enumerate(order):
    if not any(t in n for t in ("CROTCH_P", "KNEE_P", "ANKLE_P")):
      continue
    a99 = float(np.percentile(np.abs(A[:, j]), 99))
    exc = np.degrees(a99 * scale[j])
    k = names.index(n)
    meas = np.degrees(Q[:, k].max() - Q[:, k].min())
    ref = next((v for kk, v in BWC.items() if kk in n), float("nan"))
    print(f"{n:14} {a99:8.2f} {scale[j]:9.5f} {exc:13.2f} deg {meas:15.2f} deg "
          f"{ref:6.1f} deg")
  print(f"\n|action| p99 toutes articulations : {np.percentile(np.abs(A), 99):.2f}")
  print(f"|action| max                      : {np.abs(A).max():.2f}")


if __name__ == "__main__":
  main()
