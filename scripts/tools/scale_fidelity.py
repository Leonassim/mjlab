"""Le meme reseau, trois jeux d'echelles d'action.

L'echelle d'action ne voyage PAS avec le checkpoint : play.py reconstruit la
configuration depuis le code courant, et le YAML de deploiement porte sa propre
copie a la main. Trois valeurs coexistent donc pour un seul reseau :

  train  celles du run (params/env.yaml)          -- la seule fidele
  head   celles du fichier de constantes aujourd'hui -- ce que `uv run play` applique
  deploy celles du bloc mc_rtc                     -- ce que mc_mujoco applique

  uv run python scripts/tools/scale_fidelity.py <run> <model.pt> [--cmd 0.2]
"""

import argparse, re
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
DEPLOY_YAML = "/home/lmoussafir/install/lib/mc_controller/etc/NewRLQPController.yaml"


def trained_scale(run_root: Path) -> dict:
  txt = (run_root / "params" / "env.yaml").read_text()
  i = txt.index("\nactions:")
  blk = txt[i : txt.index("\nevents:", i)]
  j = blk.index("scale:")
  out = {}
  for line in blk[j:].splitlines()[1:]:
    if not line.startswith("      "):
      break
    k, _, v = line.strip().partition(":")
    out[k] = float(v)
  return out


def deploy_scale(index: int) -> dict:
  d = yaml.safe_load(open(DEPLOY_YAML))
  return d["policies"][index]["action_scale"]


def resolve(patterns: dict, names: list[str]) -> torch.Tensor:
  v = []
  for n in names:
    hit = [x for p, x in patterns.items() if p == n or re.fullmatch(p, n)]
    assert hit, f"aucune echelle pour {n}"
    v.append(hit[-1])
  return torch.tensor(v)


def run_once(label, scale_map, args, device):
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.envs
  for k in ("standing_envs", "command_vel"):
    getattr(env_cfg, "curriculum", {}).pop(k, None)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=load_rl_cfg(TASK).clip_actions,
  )
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(load_rl_cfg(TASK)), device=device
  )
  runner.load(str(Path(args.root) / args.run / args.checkpoint),
              load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  term = env.unwrapped.command_manager.get_term("twist")
  v = torch.tensor([args.cmd, 0.0, 0.0], device=device, dtype=torch.float32)

  def _fixed(env_ids):
    term.vel_command_b[env_ids] = v
    term.is_standing_env[env_ids] = False
    term.is_heading_env[env_ids] = False

  term._resample_command = _fixed

  act = env.unwrapped.action_manager.get_term("joint_pos")
  names = list(env.unwrapped.scene["robot"].joint_names)
  order = act._joint_names if hasattr(act, "_joint_names") else names
  act._scale = resolve(scale_map, list(order)).to(device)

  robot = env.unwrapped.scene["robot"]
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
  print(f"  {label:8} avance {float(torch.stack(vx).median()):+7.4f} m/s   "
        f"tremor {float(torch.stack(tremor).mean())*1e3:7.4f} mrad   "
        f"chutes {fall:5.3f}")
  env.close()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run"); p.add_argument("checkpoint")
  p.add_argument("--cmd", type=float, default=0.2)
  p.add_argument("--steps", type=int, default=400)
  p.add_argument("--envs", type=int, default=32)
  p.add_argument("--index", type=int, default=13)
  p.add_argument("--root", default="logs/rsl_rl/rhps1_velocity")
  p.add_argument("--cpu", action="store_true")
  a = p.parse_args()
  device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0"

  head = load_env_cfg(TASK, play=True).actions["joint_pos"].scale
  sets = {
    "train": trained_scale(Path(a.root) / a.run),
    "head": dict(head),
    "deploy": deploy_scale(a.index),
  }
  print(f"\n{a.run} {a.checkpoint}   commande {a.cmd:.2f} m/s   {a.envs} envs "
        f"{a.steps} pas   ({device})")
  for k, s in sets.items():
    run_once(k, s, a, device)


if __name__ == "__main__":
  main()
