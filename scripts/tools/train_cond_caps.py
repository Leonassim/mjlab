"""Landing speed, impact peak and torque peaks of a checkpoint UNDER TRAINING CONDITIONS
(stochastic actions, random 0-45 ms delay, the plant randomization of the launch)."""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

run, ck = sys.argv[1], sys.argv[2]
N = 64
cfg = load_env_cfg(g.TASK, play=False)
cfg.scene.num_envs = N
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
runner = load_runner_cls(g.TASK)(env, asdict(load_rl_cfg(g.TASK)), device="cpu")
runner.load(
  str(Path("logs/rsl_rl/rhps1_tracking") / run / ck),
  load_cfg={"actor": True},
  strict=True,
  map_location="cpu",
)
actor = runner.alg.actor
robot, sens = u.scene["robot"], u.scene["feet_ground_contact"]
bn = list(robot.body_names)
ia = [bn.index("L_ANKLE_P_LINK"), bn.index("R_ANKLE_P_LINK")]
jn = list(robot.joint_names)
obs, _ = env.reset()
rec = {k: [] for k in ("f", "vz", "tau", "qd")}
dt = u.step_dt
with torch.no_grad():
  for _ in range(int(20.0 / dt)):
    a = actor(obs, stochastic_output=True)
    obs, _, _, _ = env.step(a)
    rec["f"].append(sens.data.force.norm(dim=-1).reshape(N, -1).numpy().copy())
    rec["vz"].append(robot.data.body_link_lin_vel_w[:, ia, 2].numpy().copy())
    rec["tau"].append(robot.data.qfrc_actuator.abs().numpy().copy())
    rec["qd"].append(robot.data.joint_vel.abs().numpy().copy())
f, vz, tau, qd = (np.stack(rec[k], 1) for k in ("f", "vz", "tau", "qd"))
settle = int(
  1.0 / dt
)  # first second after reset excluded below via episode starts unknown: use t > 1 s
f, vz, tau, qd = f[:, settle:], vz[:, settle:], tau[:, settle:], qd[:, settle:]
w50, w60 = int(0.05 / dt), int(0.06 / dt)
tdv, imp = [], []
for e in range(N):
  for s in range(2):
    on = f[e, :, s] > 30
    for t0 in np.where(np.diff(on.astype(int)) == 1)[0]:
      if t0 < w50 or t0 + w60 >= f.shape[1]:
        continue
      tdv.append(-vz[e, t0 - w50 : t0 + 1, s].min())
      imp.append(f[e, t0 : t0 + w60, s].max())
tdv, imp = np.array(tdv), np.array(imp)
ok = imp < 3000  # drop falls
print(
  f"{run} {ck} conditions d'entrainement : {len(tdv)} poses ; vitesse de pose p90 {np.percentile(tdv[ok], 90):.3f} p99 {np.percentile(tdv[ok], 99):.3f} m/s ; impact p90 {np.percentile(imp[ok], 90):.0f} p99 {np.percentile(imp[ok], 99):.0f} N"
)
for gname in ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE", "ANKLE_P", "ANKLE_R"):
  cols = [i for i, n in enumerate(jn) if gname in n]
  print(
    f"  {gname:9s} couple p90 {np.percentile(tau[:, :, cols], 90):6.1f} p99 {np.percentile(tau[:, :, cols], 99):6.1f} p99.9 {np.percentile(tau[:, :, cols], 99.9):6.1f} N.m | vitesse p90 {np.percentile(qd[:, :, cols], 90):4.2f} p99 {np.percentile(qd[:, :, cols], 99):4.2f} p99.9 {np.percentile(qd[:, :, cols], 99.9):4.2f} rad/s"
  )
