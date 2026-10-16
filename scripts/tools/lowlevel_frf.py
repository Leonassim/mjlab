"""Frequency response policy target -> joint position, in mjlab, to set against the robot's.

  uv run python scripts/tools/lowlevel_frf.py <run> <ckpt.pt> [delay_ms=10]

Walks the policy in the six directions in turn (16 envs, 4 s each, held commands)
and fits, per joint, |H| and phase at 1-8 Hz from data.joint_pos_target (the
network target, before delay, posture filter and RHPS1_LOWLEVEL_LAG) to
data.joint_pos. On the robot the same chain is NewRLQPController_RL_q -> qIn.
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

JOINTS = ["L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P",
          "CHEST_Y", "L_SHOULDER_P", "L_ELBOW_P"]
N = 16


def welch(x, y, fs, n=256):
  win = np.hanning(n)
  sxx = sxy = syy = 0
  for s in range(0, len(x) - n, n // 2):
    X = np.fft.rfft((x[s:s + n] - x[s:s + n].mean()) * win)
    Y = np.fft.rfft((y[s:s + n] - y[s:s + n].mean()) * win)
    sxx, sxy, syy = sxx + np.abs(X) ** 2, sxy + np.conj(X) * Y, syy + np.abs(Y) ** 2
  return np.fft.rfftfreq(n, 1 / fs), sxy / sxx, np.abs(sxy) ** 2 / (sxx * syy)


def main():
  run, ck = sys.argv[1], sys.argv[2]
  delay_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
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
  names = list(robot.joint_names)
  dt = u.step_dt
  dirs = list(g.CMDS.values())
  tgt, pos = [], []
  obs, _ = env.reset()
  with torch.no_grad():
    for k in range(int((2.0 + 4.0 * len(dirs)) / dt)):
      s = k * dt
      i = int((s - 2.0) // 4.0)
      cmd[:] = torch.tensor(dirs[i] if s >= 2.0 and i < len(dirs) else (0, 0, 0), dtype=torch.float32)
      obs, _, _, _ = env.step(pol(obs) if s >= 1.0 else torch.zeros(N, env.num_actions))
      if s >= 2.0:
        tgt.append(robot.data.joint_pos_target.numpy().copy())
        pos.append(robot.data.joint_pos.numpy().copy())
  tgt, pos = np.stack(tgt, 1), np.stack(pos, 1)
  fq = [1, 2, 3, 4, 5, 5.5, 6, 7, 8]
  print(f"{run} {ck}  retard {delay_ms:.0f} ms  step {1000*dt:.0f} ms  gain (phase deg) cible reseau -> position")
  print(f"{'':12s} " + " ".join(f"{x:>11g}Hz" for x in fq))
  for jn in JOINTS:
    j = names.index(jn)
    Hs = []
    for e in range(N):
      fr, H, coh = welch(tgt[e, :, j], pos[e, :, j], 1 / dt)
      Hs.append(H)
    H = np.mean(Hs, 0)
    row = [f"{np.abs(H[np.argmin(np.abs(fr - x))]):4.2f}({np.degrees(np.angle(H[np.argmin(np.abs(fr - x))])):+4.0f})" for x in fq]
    print(f"{jn:12s} " + " ".join(f"{r:>13s}" for r in row))


if __name__ == "__main__":
  main()
