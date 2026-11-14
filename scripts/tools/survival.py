"""Combien de temps la politique tient-elle debout ? En SECONDES, pas en pas.

Les clips wandb font 3 s. Le robot tombe a 4.9 s dans mc_mujoco. Aucun verdict
pris sur un clip de 3 s -- ni sur un banc de 400 pas, soit 2.0 s -- ne peut voir
cette chute. Ce script mesure la survie sur une duree qui la depasse.

  uv run python scripts/tools/survival.py <run> <model.pt> [--secs 20]
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
  p.add_argument("--cmds", default="0.30,0.20,0.00")
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

  print(f"\n{a.run} {a.checkpoint}   {a.envs} envs   {a.secs:.0f} s "
        f"({steps} pas)   ({device})", flush=True)
  print(f"  {'commande':>9} {'survie mediane':>15} {'tombes':>8} {'t 1e chute':>11} "
        f"{'avance':>9}", flush=True)

  for c in [float(x) for x in a.cmds.split(",")]:
    v = torch.tensor([c, 0.0, 0.0], device=device, dtype=torch.float32)

    def _fixed(ids):
      term.vel_command_b[ids] = v
      term.is_standing_env[ids] = False
      term.is_heading_env[ids] = False

    term._resample_command = _fixed
    obs, _ = env.reset()
    _fixed(slice(None))
    alive = torch.ones(a.envs, dtype=torch.bool, device=device)
    life = torch.full((a.envs,), float(steps), device=device)
    vx = []
    with torch.no_grad():
      for i in range(steps):
        obs, _, dones, _ = env.step(policy(obs))
        newly = alive & dones.bool()
        life[newly] = float(i)
        alive &= ~dones.bool()
        if i > steps // 20 and alive.any():
          vx.append(env.unwrapped.scene["robot"].data.root_link_lin_vel_b[alive, 0]
                    .median().cpu())
        if not alive.any():
          break
    fell = int((life < steps).sum())
    first = float(life.min()) * STEP_DT if fell else float("nan")
    med = float(life.median()) * STEP_DT
    adv = float(torch.stack(vx).median()) if vx else float("nan")
    print(f"  {c:9.2f} {med:14.1f}s {fell:4d}/{a.envs} {first:10.1f}s "
          f"{adv:+9.4f}", flush=True)


if __name__ == "__main__":
  main()
