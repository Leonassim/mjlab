"""Vitesse et force d'impact pendant une MARCHE ETABLIE.

Meme discipline que demand_walking.py : play (ni randomisation ni poussee),
commande epinglee, masque de vie excluant tout env ayant termine une fois. Sans
ca on mesure le pire robot de la population, pas la marche.

On lit les capteurs directement plutot que les Metrics/* du bareme : ces termes
ne sont pas les memes d'une ablation a l'autre, donc comparer policy 0 et 6/6
par leurs recompenses comparerait deux definitions.

  uv run python scripts/tools/impact_walking.py <run> <model.pt> [--cmd 0.2]
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
FORCE = ["robot/LeftFootForceSensor_fsensor", "robot/RightFootForceSensor_fsensor"]
VEL = ["robot/left_foot_lin_vel", "robot/right_foot_lin_vel"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--cmd", type=float, default=0.2)
  p.add_argument("--steps", type=int, default=800)
  p.add_argument("--envs", type=int, default=256)
  p.add_argument("--contact-n", type=float, default=50.0, help="seuil de contact, N")
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  a = p.parse_args()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = a.envs
  for k in ("standing_envs", "command_vel"):
    getattr(cfg, "curriculum", {}).pop(k, None)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device),
                           clip_actions=load_rl_cfg(TASK).clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(str(Path(a.root) / a.run / a.checkpoint), load_cfg={"actor": True},
              strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  term = env.unwrapped.command_manager.get_term("twist")
  v = torch.tensor([a.cmd, 0.0, 0.0], device=device, dtype=torch.float32)

  def _fixed(ids):
    term.vel_command_b[ids] = v
    term.is_standing_env[ids] = False
    term.is_heading_env[ids] = False

  term._resample_command = _fixed

  scene = env.unwrapped.scene
  try:
    weight = float(sum(b.mass for b in scene["robot"].spec.bodies)) * 9.81
  except Exception:
    weight = None

  obs, _ = env.reset(); _fixed(slice(None))
  alive = torch.ones(a.envs, dtype=torch.bool, device=device)
  prev_contact = torch.zeros(a.envs, 2, dtype=torch.bool, device=device)
  prev_vz = torch.zeros(a.envs, 2, device=device)
  land_vz, land_f = [], []
  with torch.no_grad():
    for i in range(a.steps):
      obs, _, dones, _ = env.step(policy(obs))
      alive &= ~dones.bool()
      # scene[n].data EST le tenseur (num_envs, 3), pas un objet a champs.
      f = torch.stack([scene[n].data.norm(dim=-1) for n in FORCE], dim=1)
      vz = torch.stack([scene[n].data[:, 2] for n in VEL], dim=1)
      contact = f > a.contact_n
      touchdown = contact & ~prev_contact & alive.unsqueeze(1)
      if i > a.steps // 4 and touchdown.any():
        land_vz.append((-prev_vz[touchdown]).cpu())   # descente positive
        land_f.append(f[touchdown].cpu())
      prev_contact, prev_vz = contact, vz

  print(f"\ncommande {a.cmd:.2f} m/s   vivants "
        f"{int(alive.sum())}/{a.envs}   poses detectees "
        f"{sum(x.numel() for x in land_vz)}")
  if not land_vz:
    print("aucune pose detectee"); return
  vzc = torch.cat(land_vz); fc = torch.cat(land_f)
  vzc = vzc[vzc > 0]
  q = lambda t, p: float(torch.quantile(t.float(), p))
  print(f"\nvitesse de descente a la pose (m/s)")
  print(f"  p50 {q(vzc,.5):.4f}   p90 {q(vzc,.9):.4f}   p99 {q(vzc,.99):.4f}   max {float(vzc.max()):.4f}")
  print(f"\nforce verticale au premier contact (N)")
  print(f"  p50 {q(fc,.5):7.1f}   p90 {q(fc,.9):7.1f}   p99 {q(fc,.99):7.1f}   max {float(fc.max()):7.1f}")
  if weight:
    print(f"  en poids du robot ({weight:.0f} N) : p50 {q(fc,.5)/weight:.2f}x   "
          f"p99 {q(fc,.99)/weight:.2f}x   max {float(fc.max())/weight:.2f}x")


if __name__ == "__main__":
  main()
