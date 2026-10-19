"""Dynamic envelope of a checkpoint: what a new gait must not exceed.

  uv run python scripts/tools/dyn_envelope.py <run> <ckpt.pt> [<run> <ckpt.pt> ...]

The first checkpoint is the reference (slot 13 = the BWC copy the robot already
walks with); the others are printed as ratios to it. Six directions x 8 envs,
30 ms delay, filtered command, plant from the environment, 10 s of walking each
after a standing start. Per joint group: actuator torque p99 and max (and the
max against the effort limit), joint speed p99; per foot: touchdown impact (peak
normal force in the 60 ms after touchdown, p99) and stance slip (foot speed while
loaded above 100 N, p90).
"""

import os
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

CMDS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
PER = 8
GROUPS = ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE", "ANKLE_P", "ANKLE_R", "CHEST", "SHOULDER", "ELBOW")


def rollout(run_dir, ck):
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
  target = torch.tensor([c for c in CMDS for _ in range(PER)], dtype=torch.float32)
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
  runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run_dir / ck), load_cfg={"actor": True},
              strict=True, map_location="cpu")
  pol = runner.get_inference_policy(device="cpu")
  robot = u.scene["robot"]
  sens = u.scene["feet_ground_contact"]
  bn = list(robot.body_names)
  ia = [bn.index("L_ANKLE_P_LINK"), bn.index("R_ANKLE_P_LINK")]
  rec = {k: [] for k in ("tau", "qd", "f", "v")}
  obs, _ = env.reset()
  dt = u.step_dt
  with torch.no_grad():
    for k in range(int(16.0 / dt)):
      cmd[:] = target if k * dt >= 3.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
      if k * dt >= 6.0:
        d = robot.data
        rec["tau"].append(d.qfrc_actuator.numpy().copy())
        rec["qd"].append(d.joint_vel.numpy().copy())
        rec["f"].append(sens.data.force.norm(dim=-1).reshape(N, -1).numpy().copy())
        rec["v"].append(d.body_link_lin_vel_w[:, ia, :2].norm(dim=-1).numpy().copy())
  r = {k: np.stack(v, 1) for k, v in rec.items()}  # (N, T, ...)
  jn = list(robot.joint_names)
  lim = {}
  for a in cfg.scene.entities["robot"].articulation.actuators:
    for gname in GROUPS:
      if any(gname in e for e in a.target_names_expr):
        lim[gname] = a.effort_limit
  out = {}
  for gname in GROUPS:
    cols = [i for i, n in enumerate(jn) if gname in n]
    t = np.abs(r["tau"][:, :, cols]).reshape(-1)
    q = np.abs(r["qd"][:, :, cols]).reshape(-1)
    out[gname] = (np.percentile(t, 99), t.max(), lim.get(gname, np.nan), np.percentile(q, 99))
  f, v = r["f"], r["v"]
  imp, slip = [], []
  w = int(0.06 / dt)
  for e in range(f.shape[0]):
    for s in range(2):
      on = f[e, :, s] > 100
      for t0 in np.where(np.diff(on.astype(int)) == 1)[0]:
        imp.append(f[e, t0:t0 + w, s].max())
      slip.extend(v[e, on, s])
  out["impact"] = np.percentile(imp, 99) if imp else np.nan
  out["slip"] = np.percentile(slip, 90) if slip else np.nan
  if os.environ.get("DYN_PER_CMD") == "1":  # where the peaks come from
    for ci, c in enumerate(CMDS):
      es = slice(ci * PER, (ci + 1) * PER)
      row = []
      for gname in ("KNEE", "CROTCH_R", "CROTCH_P", "ANKLE_R", "ANKLE_P"):
        cols = [i for i, n in enumerate(jn) if gname in n]
        t = np.abs(r["tau"][es][:, :, cols])
        row.append(f"{gname} p99 {np.percentile(t, 99):5.1f} max {t.max():5.1f}")
      fi = [f[e, t0:t0 + w, s].max() for e in range(es.start, es.stop) for s in range(2)
            for t0 in np.where(np.diff((f[e, :, s] > 100).astype(int)) == 1)[0]]
      print(f"  cmd {str(c):12s} " + " | ".join(row) + f" | impact p99 {np.percentile(fi, 99) if fi else np.nan:4.0f} N")
  return out


def main():
  args = sys.argv[1:]
  ref = None
  for a in range(0, len(args), 2):
    o = rollout(args[a], args[a + 1])
    print(f"== {args[a]} {args[a + 1]}" + ("" if ref is None else "   (rapport a la reference)"))
    for gname in GROUPS:
      p99, mx, lim, qd = o[gname]
      line = f"  {gname:9s} couple p99 {p99:6.1f} max {mx:6.1f} N.m (limite {lim:5.0f}, max/limite {mx / lim:4.2f}) | vitesse p99 {qd:4.1f} rad/s"
      if ref is not None:
        line += f"   x{p99 / ref[gname][0]:.2f} / x{mx / ref[gname][1]:.2f} / x{qd / ref[gname][3]:.2f}"
      print(line)
    line = f"  impact pied p99 {o['impact']:5.0f} N | glissement appui p90 {o['slip'] * 100:4.1f} cm/s"
    if ref is not None:
      line += f"   x{o['impact'] / ref['impact']:.2f} / x{o['slip'] / ref['slip']:.2f}"
    print(line, flush=True)
    if ref is None:
      ref = o


if __name__ == "__main__":
  main()
