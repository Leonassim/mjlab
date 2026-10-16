"""Does the policy keep vibrating at rest after a walk, as the robot does?

  uv run python scripts/tools/rest_vibration.py <delay_ms> <run> <ckpt.pt> [<run> <ckpt.pt> ...]

16 envs: 2 s standing, 6 s walking (4 directions, 4 envs each), then 12 s at
command 0. Over the last 10 s: base yaw-rate rms (the real robot: 4.6 deg/s on
slot 9, 9.9 on slot 10, at 5.5 Hz) and the dominant frequency of the policy
target. RHPS1_LOWLEVEL_LAG / RHPS1_ROBUST from the environment as usual.
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

N = 16
DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1)]


def rollout(delay_ms, run, ck):
  cfg = load_env_cfg(g.TASK, play=True)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = g.MOTION
  lag = int(round(delay_ms / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
  cmd = torch.zeros(N, 3)
  walk = torch.tensor([DIRS[i // 4] for i in range(N)], dtype=torch.float32)
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
  wz, tg, z = [], [], []
  obs, _ = env.reset()
  with torch.no_grad():
    for k in range(int(20.0 / dt)):
      s = k * dt
      cmd[:] = walk if 2.0 <= s < 8.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if s >= 1.0 else torch.zeros(N, env.num_actions))
      if s >= 10.0:
        wz.append(robot.data.root_link_ang_vel_b[:, 2].numpy().copy())
        tg.append(robot.data.joint_pos_target.numpy().copy())
        z.append(robot.data.root_link_pos_w[:, 2].numpy().copy())
  return np.stack(wz, 1), np.stack(tg, 1), np.stack(z, 1), dt


if __name__ == "__main__":
  delay_ms = float(sys.argv[1])
  args = sys.argv[2:]
  for run, ck in zip(args[::2], args[1::2]):
    wz, tg, z, dt = rollout(delay_ms, run, ck)
    up = z.min(1) > 0.6
    rms = np.degrees(np.sqrt((wz ** 2).mean(1)))
    x = tg[up] - tg[up].mean(1, keepdims=True)
    F = np.abs(np.fft.rfft(x, axis=1)).sum((0, 2))
    fr = np.fft.rfftfreq(x.shape[1], dt)
    sel = fr > 1.0
    print(f"{run} {ck} retard {delay_ms:.0f} ms: lacet bassin rms median {np.median(rms[up]):.2f} deg/s "
          f"(max {rms[up].max():.2f}), pic cible {fr[sel][np.argmax(F[sel])]:.1f} Hz, debout {up.sum()}/{N}")
