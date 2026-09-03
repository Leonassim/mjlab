"""Dimensionner torque_demand_overshoot sur une politique existante.

Regle M1 : mesurer la valeur BRUTE du terme avant de lui donner un poids. Trois
termes de couple ont deja ete regles a l'aveugle sur joint_torque_limit_margin,
qui lit data.actuator_force -- borne par construction, ratio <= 1.0 impossible a
depasser. Deviner un poids par analogie avec ce terme reproduirait l'erreur.

  uv run python scripts/tools/size_demand_penalty.py <run> <model_xxx.pt>

Conditions d'ENTRAINEMENT (randomisation active), pas play : le poids sert a
l'entrainement, le mesurer sans randomisation donnerait la mauvaise echelle.
"""

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run")
  p.add_argument("checkpoint")
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--envs", type=int, default=512)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  a = p.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK, play=False)
  env_cfg.scene.num_envs = a.envs
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=load_rl_cfg(TASK).clip_actions,
  )
  agent_cfg = load_rl_cfg(TASK)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(agent_cfg), device=device
  )
  runner.load(str(Path(a.root) / a.run / a.checkpoint), load_cfg={"actor": True},
              strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]
  acts = [x for x in robot.actuators if hasattr(x, "_raw_torque_peak")]
  names = list(robot.joint_names)
  assert acts, "aucun FiniteDifferencePdActuator"

  raw, ratio_max, over_frac = [], [], []
  per_joint = torch.zeros(len(names), device=device)
  pj_max = torch.zeros(len(names), device=device)
  pj_over = torch.zeros(len(names), device=device)
  per_joint_n = 0
  obs, _ = env.reset()
  with torch.no_grad():
    for i in range(a.steps):
      obs, _, _, _ = env.step(policy(obs))
      r = [x._raw_torque_peak for x in acts if x._raw_torque_peak is not None]
      if not r:
        continue
      ratio = torch.cat(r, dim=1)
      excess = torch.clamp(ratio - 1.0, min=0.0)
      if i > a.steps // 4:
        raw.append(float(torch.sum(torch.square(excess), dim=1).mean()))
        ratio_max.append(float(ratio.max()))
        over_frac.append(float((ratio > 1.0).float().mean()))
        if ratio.shape[1] == len(names):
          per_joint += torch.square(excess).mean(dim=0)
          pj_max = torch.maximum(pj_max, ratio.max(dim=0).values)
          pj_over += (ratio > 1.0).float().mean(dim=0)
          per_joint_n += 1

  raw.sort()
  print(f"\nvaleur BRUTE du terme, somme des exces au carre, par pas de politique")
  print(f"  moyenne          {statistics.fmean(raw):10.4f}")
  print(f"  mediane          {raw[len(raw)//2]:10.4f}")
  print(f"  p90              {raw[int(0.90*len(raw))]:10.4f}")
  print(f"  p99              {raw[int(0.99*len(raw))]:10.4f}")
  print(f"  max              {raw[-1]:10.4f}")
  print(f"\nratio de demande   max {max(ratio_max):.2f}   "
        f"fraction au-dessus {statistics.fmean(over_frac):.4f}")
  if per_joint_n:
    pj = (per_joint / per_joint_n).cpu()
    pm = pj_max.cpu()
    po = (pj_over / per_joint_n).cpu()
    order = torch.argsort(pj, descending=True)
    tot = float(pj.sum())
    print(f"\ntoutes les articulations   (total exces^2 = {tot:.4f})")
    print(f"  {'joint':16s} {'exces^2':>10s} {'part':>7s} {'ratioMax':>9s} {'fracOver':>9s}")
    for k in order:
      k = int(k)
      print(f"  {names[k]:16s} {float(pj[k]):10.5f} {100*float(pj[k])/max(tot,1e-9):6.1f}% "
            f"{float(pm[k]):9.2f} {float(po[k]):9.4f}")


if __name__ == "__main__":
  main()
