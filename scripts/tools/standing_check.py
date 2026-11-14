"""Que fait la politique a COMMANDE NULLE, sur une duree qui depasse le clip.

Leo, 2026-09-10 : "elle fait des petits pas a zero vitesse alors qu'elle
devrait etre parfaitement stable". Ce script chiffre exactement ca : poses de
pied par seconde, part du temps sur un pied, et vitesse articulaire des jambes
-- la grandeur que standing_joint_vel taxe, pour savoir si ce terme peut mordre.

  uv run python scripts/tools/standing_check.py <run> <model.pt> [--secs 20]
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
STEP_DT = 0.005


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--secs", type=float, default=20.0)
  p.add_argument("--envs", type=int, default=64)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  p.add_argument("--cpu", action="store_true")
  a = p.parse_args()
  device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0"
  steps = int(a.secs / STEP_DT)

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
  zero = torch.zeros(3, device=device)

  def _fixed(ids):
    term.vel_command_b[ids] = zero
    term.is_standing_env[ids] = False
    term.is_heading_env[ids] = False

  term._resample_command = _fixed
  robot = env.unwrapped.scene["robot"]
  names = list(robot.joint_names)
  leg = [i for i, n in enumerate(names)
         if any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]
  sensor = env.unwrapped.scene["feet_ground_contact"]

  obs, _ = env.reset()
  _fixed(slice(None))
  alive = torch.ones(a.envs, dtype=torch.bool, device=device)
  prev_contact = None
  td = torch.zeros(a.envs, device=device)
  one_foot, two_feet, qd_abs, qd_sq, n = 0.0, 0.0, 0.0, 0.0, 0
  with torch.no_grad():
    for i in range(steps):
      obs, _, dones, _ = env.step(policy(obs))
      alive &= ~dones.bool()
      if i <= steps // 10 or not alive.any():
        prev_contact = None
        continue
      f = sensor.data.current_contact_time > 0.0        # [B, n_geoms]
      # regrouper par pied : moities gauche/droite des geometries
      h = f.shape[1] // 2
      c = torch.stack([f[:, :h].any(1), f[:, h:].any(1)], dim=1).float()
      nfeet = c.sum(1)
      one_foot += float((nfeet[alive] == 1).float().mean()); two_feet += float((nfeet[alive] == 2).float().mean())
      if prev_contact is not None:
        td += ((c == 1) & (prev_contact == 0)).any(1).float()
      prev_contact = c
      qd = robot.data.joint_vel[:, leg][alive]
      qd_abs += float(qd.abs().mean())
      qd_sq += float(torch.square(robot.data.joint_vel[alive]).sum(1).mean())
      n += 1

  dur = n * STEP_DT
  print(f"\n{a.run} {a.checkpoint}   commande NULLE   {a.envs} envs   {dur:.1f} s")
  print(f"  poses de pied           {float(td[alive].mean())/dur:7.2f} /s")
  print(f"  temps sur UN pied       {one_foot/n:7.3f}")
  print(f"  temps sur DEUX pieds    {two_feet/n:7.3f}")
  print(f"  |qd| jambes moyen       {qd_abs/n:7.4f} rad/s")
  print(f"  sum(qd^2) 30 joints     {qd_sq/n:7.4f}   <- ce que standing_joint_vel taxe")
  print(f"  chutes                  {1.0 - float(alive.float().mean()):7.3f}")


if __name__ == "__main__":
  main()
