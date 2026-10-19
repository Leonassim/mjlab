"""Write mjlab rollouts in real_steps.py's npz format, to measure sim like the robot.

  uv run python scripts/tools/sim_steps.py <run> <ckpt.pt> <out_prefix>
  uv run python scripts/tools/real_steps.py compliance <out_prefix>_*.npz

One npz per command (yaw right, lateral right, forward; 6 envs each, 30 ms
delay, filtered command, plant from the environment: RHPS1_FOOT_SOLREF,
RHPS1_LOWLEVEL_LAG...). Envs are concatenated with an unarmed sample between
them, so no single-support phase spans two envs.
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

CMDS = {"yawR": (0, 0, -1), "latR": (0, -1, 0), "fwd": (1, 0, 0)}
PER = 6
J = ("CHEST_Y,CHEST_P,HEAD_Y,HEAD_P,L_SHOULDER_P,L_SHOULDER_R,L_SHOULDER_Y,L_ELBOW_P,L_ELBOW_Y,"
     "L_WRIST_R,L_WRIST_Y,R_SHOULDER_P,R_SHOULDER_R,R_SHOULDER_Y,R_ELBOW_P,R_ELBOW_Y,R_WRIST_R,"
     "R_WRIST_Y,L_CROTCH_Y,L_CROTCH_R,L_CROTCH_P,L_KNEE_P,L_ANKLE_R,L_ANKLE_P,R_CROTCH_Y,R_CROTCH_R,"
     "R_CROTCH_P,R_KNEE_P,R_ANKLE_R,R_ANKLE_P").split(",")


def main():
  run, ck, out = sys.argv[1], sys.argv[2], sys.argv[3]
  N = PER * len(CMDS)
  cfg = load_env_cfg(g.TASK, play=True)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = g.MOTION
  lag = int(round(30 / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
  target = torch.tensor([c for c in CMDS.values() for _ in range(PER)], dtype=torch.float32)
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
  sens = u.scene["feet_ground_contact"]
  jidx = [list(robot.joint_names).index(n) for n in J]
  rec = {k: [] for k in ("q", "grav", "f", "cmd", "quat", "gyro")}
  obs, _ = env.reset()
  dt = u.step_dt
  with torch.no_grad():
    for k in range(int(16.0 / dt)):
      cmd[:] = target if k * dt >= 3.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
      if k * dt >= 5.0:
        qq = robot.data.joint_pos.clone()
        if hasattr(robot, "_rhps1_deflection"):
          qq = qq + robot._rhps1_deflection  # encoder side, as on the robot
        rec["q"].append(qq[:, jidx].numpy().copy())
        rec["grav"].append(robot.data.projected_gravity_b.numpy().copy())
        rec["gyro"].append(robot.data.root_link_ang_vel_b.numpy().copy())
        rec["f"].append(sens.data.force.norm(dim=-1).reshape(N, -1).numpy().copy())
        rec["cmd"].append(target.numpy().astype(int))
        qq = robot.data.root_link_quat_w.numpy().copy()
        rec["quat"].append(qq * np.array([1.0, -1.0, -1.0, -1.0]))  # stored inverted, as mc_rtc logs it
  rec = {k: np.stack(v, 1) for k, v in rec.items()}  # (N, T, ...)
  for i, name in enumerate(CMDS):
    blocks = {k: [] for k in ("q", "grav", "forces", "cmd", "armed", "quat", "gyro")}
    for e in range(i * PER, (i + 1) * PER):
      T = rec["q"].shape[1]
      blocks["q"] += [rec["q"][e], np.zeros((1, 30))]
      blocks["grav"] += [rec["grav"][e], np.zeros((1, 3))]
      blocks["forces"] += [rec["f"][e], np.zeros((1, 2))]
      blocks["cmd"] += [rec["cmd"][e], np.zeros((1, 3), dtype=int)]
      blocks["armed"] += [np.ones(T, dtype=bool), np.zeros(1, dtype=bool)]
      blocks["quat"] += [rec["quat"][e], np.array([[1.0, 0, 0, 0]])]
      blocks["gyro"] += [rec["gyro"][e], np.zeros((1, 3))]
    np.savez_compressed(f"{out}_{name}.npz", **{k: np.concatenate(v) for k, v in blocks.items()})


if __name__ == "__main__":
  main()
