"""Demande de couple pendant une MARCHE ETABLIE, pas pendant une chute.

size_demand_penalty.py mesurait en conditions d'entrainement -- randomisation,
poussees, commandes tirees au hasard -- et rapportait le maximum sur 512 robots
dont certains tombaient. Le "ratio max 9.14" de la policy 0 etait donc peut-etre
le pic d'un robot en train de tomber, ce que mc_mujoco ne reproduit pas.

Trois differences, et chacune corrige une facon de mentir :

  play=True         pas de randomisation ni de poussees, ce qu'est mc_mujoco
  commande epinglee une vitesse de marche fixe, pas le curriculum
  masque de vie     un env qui a termine une fois est exclu pour de bon

Et on rapporte des PERCENTILES, pas seulement le maximum : sur 512 envs x 300
pas, le max est un echantillon sur 150 000, c'est-a-dire la queue.

  uv run python scripts/tools/demand_walking.py <run> <model.pt> [--cmd 0.2]
"""

import argparse
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
  p.add_argument("--cmd", type=float, default=0.2, help="vitesse avant, m/s")
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--envs", type=int, default=256)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  a = p.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = a.envs
  for k in ("standing_envs", "command_vel"):
    getattr(env_cfg, "curriculum", {}).pop(k, None)

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=load_rl_cfg(TASK).clip_actions,
  )
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(load_rl_cfg(TASK)), device=device
  )
  runner.load(str(Path(a.root) / a.run / a.checkpoint), load_cfg={"actor": True},
              strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  # Epingler la commande. Le terme se re-echantillonne sur son propre minuteur,
  # donc une valeur poussee de l'exterieur est remplacee une seconde plus tard.
  term = env.unwrapped.command_manager.get_term("twist")
  v = torch.tensor([a.cmd, 0.0, 0.0], device=device, dtype=torch.float32)

  def _fixed(env_ids):
    term.vel_command_b[env_ids] = v
    term.is_standing_env[env_ids] = False
    term.is_heading_env[env_ids] = False

  term._resample_command = _fixed

  robot = env.unwrapped.scene["robot"]
  acts = [x for x in robot.actuators if hasattr(x, "_raw_torque_peak")]
  names = list(robot.joint_names)
  assert acts, "aucun FiniteDifferencePdActuator"

  obs, _ = env.reset()
  _fixed(slice(None))
  alive = torch.ones(a.envs, dtype=torch.bool, device=device)
  samples = []
  with torch.no_grad():
    for i in range(a.steps):
      obs, _, dones, extras = env.step(policy(obs))
      # Un env qui a termine une fois est exclu DEFINITIVEMENT : apres un
      # reset il repart d'une pose initiale, ce qui n'est pas de la marche.
      alive &= ~dones.bool()
      r = [x._raw_torque_peak for x in acts if x._raw_torque_peak is not None]
      if not r or i <= a.steps // 4:
        continue
      ratio = torch.cat(r, dim=1)
      if ratio.shape[1] == len(names) and alive.any():
        samples.append(ratio[alive].cpu())

  print(f"\ncommande {a.cmd:.2f} m/s   envs vivants {int(alive.sum())}/{a.envs}"
        f"   echantillons {sum(s.shape[0] for s in samples)}")
  if not samples:
    print("aucun echantillon : tous les envs ont termine"); return
  r = torch.cat(samples, dim=0)
  excess = torch.clamp(r - 1.0, min=0.0)
  print(f"total exces^2 moyen par pas   {float(torch.square(excess).sum(1).mean()):.4f}")
  print(f"ratio max global              {float(r.max()):.2f}")
  print(f"fraction au-dessus            {float((r > 1.0).float().mean()):.4f}\n")

  pj = torch.square(excess).mean(0)
  order = torch.argsort(pj, descending=True)
  print(f"  {'joint':16s} {'exces^2':>9s} {'p50':>7s} {'p99':>7s} {'p999':>7s} {'max':>7s} {'frac>1':>8s}")
  for k in order[:14]:
    k = int(k)
    c = r[:, k]
    q = torch.quantile(c.float(), torch.tensor([0.5, 0.99, 0.999]))
    print(f"  {names[k]:16s} {float(pj[k]):9.5f} {float(q[0]):7.2f} {float(q[1]):7.2f} "
          f"{float(q[2]):7.2f} {float(c.max()):7.2f} {float((c > 1).float().mean()):8.4f}")


if __name__ == "__main__":
  main()
