"""Chaque ecart de deploiement, applique dans mjlab, sur le meme reseau.

wandb montre une marche, `uv run play` et mc_mujoco non. Le plant d'entrainement
n'est reproduit ni par l'un ni par l'autre, et pour trois raisons distinctes :

  damper   velocity_damper_progress est monte a 1.0 par un CURRICULUM, qui ne
           tourne qu'a l'entrainement. En lecture il vaut 0, et le YAML de
           deploiement portait velocity_damper_di: 0.0 -- les deux ont donc le
           damper desarme alors que l'entrainement l'avait plein.
  clamp    mjlab borne le couple PD a effort_limit cote Python. mc_mujoco ne
           borne rien (MjRobot::PD, XML forcelimited="false").
  echelles le YAML portait les echelles arrondies des metadonnees ONNX.

Chaque condition est isolee, puis les trois ensemble.

  uv run python scripts/tools/plant_deviation.py <run> <model.pt> [--cmd 0.2]
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.actuator.finite_difference_pd_actuator import FiniteDifferencePdActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"

# Ce que l'entrainement avait vraiment au checkpoint. progress=1.0 depuis
# l'iteration 12750 (curriculum 360000 -> 612000 pas, 48 pas/env).
TRAINED_DAMPER = 1.0


def build(args, device):
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = args.envs
  # velocity_damper part AUSSI : le curriculum tourne a chaque reset et
  # recalcule progress depuis common_step_counter, qui repart de zero en
  # lecture. Il ecraserait la valeur posee plus bas des la premiere terminaison
  # -- et c'est exactement pour cette raison que `uv run play` tourne sans
  # damper alors que l'entrainement l'avait plein.
  for k in ("standing_envs", "command_vel", "velocity_damper"):
    getattr(cfg, "curriculum", {}).pop(k, None)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device=device),
    clip_actions=load_rl_cfg(TASK).clip_actions,
  )
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(load_rl_cfg(TASK)), device=device
  )
  runner.load(str(Path(args.root) / args.run / args.checkpoint),
              load_cfg={"actor": True}, strict=True, map_location=device)
  return env, runner.get_inference_policy(device=device)


def run(label, args, device, damper, clamp, scale_mul):
  env, policy = build(args, device)
  robot = env.unwrapped.scene["robot"]
  acts = [a for a in robot.actuators if isinstance(a, FiniteDifferencePdActuator)]
  for a in acts:
    a.velocity_damper_progress = damper
    if not clamp and a.force_limit is not None:
      a.force_limit = torch.full_like(a.force_limit, 1e9)
  if scale_mul != 1.0:
    term = env.unwrapped.action_manager.get_term("joint_pos")
    term._scale = term._scale * scale_mul

  term = env.unwrapped.command_manager.get_term("twist")
  v = torch.tensor([args.cmd, 0.0, 0.0], device=device, dtype=torch.float32)

  def _fixed(ids):
    term.vel_command_b[ids] = v
    term.is_standing_env[ids] = False
    term.is_heading_env[ids] = False

  term._resample_command = _fixed
  names = list(robot.joint_names)
  leg = [i for i, n in enumerate(names)
         if any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]

  obs, _ = env.reset()
  _fixed(slice(None))
  alive = torch.ones(args.envs, dtype=torch.bool, device=device)
  vx, tremor, prev, prev_d = [], [], None, None
  with torch.no_grad():
    for i in range(args.steps):
      obs, _, dones, _ = env.step(policy(obs))
      alive &= ~dones.bool()
      if i <= args.steps // 4 or not alive.any():
        prev = prev_d = None
        continue
      q = robot.data.joint_pos[:, leg]
      if prev is not None:
        d = q - prev
        if prev_d is not None:
          tremor.append((d - prev_d)[alive].abs().mean().cpu())
        prev_d = d
      prev = q
      vx.append(robot.data.root_link_lin_vel_b[alive, 0].median().cpu())
  fall = 1.0 - float(alive.float().mean())
  a = float(torch.stack(vx).median()) if vx else float("nan")
  t = float(torch.stack(tremor).mean()) * 1e3 if tremor else float("nan")
  print(f"  {label:22} avance {a:+7.4f} m/s   tremor {t:7.4f} mrad   chutes {fall:5.3f}",
        flush=True)
  env.close()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--cmd", type=float, default=0.2)
  p.add_argument("--steps", type=int, default=400)
  p.add_argument("--envs", type=int, default=32)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  p.add_argument("--cpu", action="store_true")
  a = p.parse_args()
  device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0"
  print(f"\n{a.run} {a.checkpoint}   commande {a.cmd:.2f} m/s   {a.envs} envs "
        f"{a.steps} pas   ({device})", flush=True)
  # damper, clamp de couple, multiplicateur d'echelle
  run("plant d'entrainement", a, device, TRAINED_DAMPER, True, 1.0)
  run("damper desarme", a, device, 0.0, True, 1.0)
  run("sans clamp de couple", a, device, TRAINED_DAMPER, False, 1.0)
  run("deploiement (les 3)", a, device, 0.0, False, 1.0)


if __name__ == "__main__":
  main()
