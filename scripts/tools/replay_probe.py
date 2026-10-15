"""Replay a real-robot command sequence in mjlab: does the policy fall here too?

  uv run python scripts/tools/replay_probe.py <scenario> <run> <ckpt.pt> [<run> <ckpt.pt> ...]

Scenarios are the joystick timelines of the real tests, in seconds after
arming, as sign commands (vx, vy, wz). 16 envs, 30 ms delay, command filter
0.3 s, RHPS1_ROBUST perturbations active. A fall is a base below 0.55 m or a
tilt above 35 deg; its time and the command then are reported.
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

SCENARIOS = {
  # slot 10, 2026-10-15 exp 1: forward, stop, backward, stop -> fell forward
  "exp1": [(2.4, (1, 0, 0)), (5.2, (0, 0, 0)), (9.9, (-1, 0, 0)), (12.8, (0, 0, 0)), (22.0, None)],
  # slot 10, 2026-10-15 exp 2: yaw right 15 s, stop, diagonal forward-right, stop -> fell backward
  "exp2": [(2.0, (0, 0, -1)), (17.2, (0, 0, 0)), (26.5, (1, -1, 0)), (28.5, (0, 0, 0)), (36.0, None)],
  # the same diagonal with the dominant axis only
  "exp2_axis": [(2.0, (0, 0, -1)), (17.2, (0, 0, 0)), (26.5, (1, 0, 0)), (28.5, (0, 0, 0)), (36.0, None)],
}
N = 16


def command_at(sc, s):
  c = (0, 0, 0)
  for t0, v in sc:
    if s >= t0 and v is not None:
      c = v
  return c


def rollout(sc, run, ck):
  cfg = load_env_cfg(g.TASK, play=True)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = g.MOTION
  m = cfg.commands["motion"]
  m.sampling_mode = "start"
  m.pose_range = {}
  m.velocity_range = {}
  if hasattr(m, "standing_start_prob"):
    m.standing_start_prob = 0.0
  lag = int(round(30 / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
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
  robot = u.scene["robot"]
  dt = u.step_dt
  fell = np.full(N, np.nan)
  obs, _ = env.reset()
  arm = 1.0
  with torch.no_grad():
    for k in range(int((sc[-1][0] + arm) / dt)):
      s = k * dt - arm
      cmd[:] = torch.tensor(command_at(sc, s), dtype=torch.float32)
      obs, _, _, _ = env.step(pol(obs) if s >= 0 else torch.zeros(N, env.num_actions))
      z = robot.data.root_link_pos_w[:, 2].numpy()
      gz = robot.data.projected_gravity_b[:, 2].numpy()
      bad = (z < 0.55) | (np.degrees(np.arccos(np.clip(-gz, -1, 1))) > 35)
      new = bad & np.isnan(fell)
      fell[new] = s
  return fell


if __name__ == "__main__":
  sc = SCENARIOS[sys.argv[1]]
  args = sys.argv[2:]
  for run, ck in zip(args[::2], args[1::2]):
    fell = rollout(sc, run, ck)
    n = int(np.sum(~np.isnan(fell)))
    when = ", ".join(f"{x:.1f}s {command_at(sc, x)}" for x in np.sort(fell[~np.isnan(fell)])[:6])
    print(f"{sys.argv[1]} {run} {ck}: chutes {n}/{N}  {when}")
