"""Friction a policy uses: |F_tangential| / F_normal of each stance foot.

  uv run python scripts/tools/friction_usage.py <run> <ckpt.pt> [<run> <ckpt.pt> ...]

Same rollout as gait_dynamics (4 envs per direction, 30 ms delay, filtered
command, flat ground at RHPS1_FOOT_FRICTION). The BWC caps its own contact
wrenches at fricCoeff 0.5; a policy above that relies on grip the BWC never
asks for. Net foot force in the world frame; a foot counts as in stance above
20 % of body weight.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402


def rollout(run, ck):
  cfg = load_env_cfg(g.TASK, play=True)
  N = g.PER * len(g.CMDS)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = g.MOTION
  lag = int(round(30 / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
  target = torch.tensor([c for c in g.CMDS.values() for _ in range(g.PER)], dtype=torch.float32)
  cmd = torch.zeros(N, 3)
  filt = {"y": torch.zeros(N, 3), "step": -1}

  def _cmd(env, command_name, tau=0.0):
    step = int(env.common_step_counter)
    if filt["step"] != step:
      filt["y"] = filt["y"] + (env.step_dt / 0.3) * (cmd - filt["y"])
      filt["step"] = step
    return filt["y"].to(env.device)

  cfg.observations["actor"].terms["velocity_command"].func = _cmd
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
  u = env.unwrapped
  runner = load_runner_cls(g.TASK)(env, asdict(load_rl_cfg(g.TASK)), device="cpu")
  runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True},
              strict=True, map_location="cpu")
  pol = runner.get_inference_policy(device="cpu")
  sens = u.scene["feet_ground_contact"]
  f = []
  obs, _ = env.reset()
  with torch.no_grad():
    for k in range(int(20.0 / u.step_dt)):
      cmd[:] = target if k * u.step_dt >= 3.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if k * u.step_dt >= 1.0 else torch.zeros(N, env.num_actions))
      if k * u.step_dt >= 6.0:
        f.append(sens.data.force.reshape(N, -1, 3).numpy().copy())
  return np.stack(f, 1)  # (N, T, feet, 3)


if __name__ == "__main__":
  args = sys.argv[1:]
  w = g.MASS * g.G
  print(f"{'':28s} {'cmd':8s} {'p50':>5s} {'p95':>5s} {'p99':>5s}  {'>0.3':>5s} {'>0.4':>5s} {'>0.5':>5s}  (% stance)")
  for run, ck in zip(args[::2], args[1::2]):
    f = rollout(run, ck)
    for i, name in enumerate(g.CMDS):
      ff = f[i * g.PER:(i + 1) * g.PER].reshape(-1, 3)
      fn = -ff[:, 2]  # the sensor reports the force on the ground side: normal is -z
      st = fn > 0.2 * w
      r = np.linalg.norm(ff[st, :2], axis=1) / fn[st]
      p = np.percentile(r, [50, 95, 99])
      print(f"{run[-8:] + ' ' + ck:28s} {name:8s} {p[0]:5.2f} {p[1]:5.2f} {p[2]:5.2f}  "
            f"{100 * (r > 0.3).mean():5.1f} {100 * (r > 0.4).mean():5.1f} {100 * (r > 0.5).mean():5.1f}")
