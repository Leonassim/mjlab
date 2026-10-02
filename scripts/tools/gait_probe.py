"""Forward walk from standing: per-foot lift, swing, step count, drift.

Settle 3 s with actions held, stand 2 s at command 0, then command forward for
12 s (through the controller's filter when RHPS1_CMD_TAU is set). 16 envs started
apart. Reports per foot the median peak lift above its standing height, swing
time and step count, plus advance, lateral and yaw drift.

  uv run python scripts/tools/gait_probe.py <run> <model.pt>
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
N = 16
torch.manual_seed(0)
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
  cfg.terminations.pop(k, None)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v2.npz"
lag = int(round(float(os.environ.get("FLIP_DELAY_MS", "0")) / 1000 / cfg.sim.mujoco.timestep))
for a in cfg.scene.entities["robot"].articulation.actuators:
  a.delay_min_lag = lag
  a.delay_max_lag = lag
CMD = torch.zeros(N, 3)
TAU = float(os.environ.get("RHPS1_CMD_TAU", "0"))
_f = {"y": torch.zeros(N, 3), "step": -1}


def _cmd(env, command_name, tau=0.0):
  if TAU <= 0:
    return CMD.clone()
  s = int(env.common_step_counter)
  if _f["step"] != s:
    _f["y"] = _f["y"] + (env.step_dt / TAU) * (CMD - _f["y"])
    _f["step"] = s
  return _f["y"].clone()


cfg.observations["actor"].terms["velocity_command"].func = _cmd
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True},
            strict=True, map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
robot = u.scene["robot"]
sens = u.scene["feet_ground_contact"]
feet = [robot.body_names.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
dt = u.step_dt

obs, _ = env.reset()
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(q0 + (torch.rand_like(q0) * 2 - 1) * 0.01, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)


def tilt_deg(quat):
  """Angle between the foot's z axis and the world vertical, from wxyz quaternions."""
  w, x, y = quat[..., 0], quat[..., 1], quat[..., 2]
  zz = 1 - 2 * (x * x + y * y)  # z component of the body z axis
  return np.degrees(np.arccos(np.clip(zz, -1, 1)))


def gait(Zf, XYf, TILT, F=None):
  """Per foot, from height above stance (swing = > 1 cm, same rule for BWC and policy):
  steps, swing time, lift, step length, foot tilt at touchdown and in stance.
  Zf (T, E, 2) m, XYf (T, E, 2, 2) m, TILT (T, E, 2) deg. F forces, for double support."""
  out = []
  for k in range(2):
    st_n, sw, lift, length, td, stance = [], [], [], [], [], []
    for e in range(Zf.shape[1]):
      up = (Zf[:, e, k] > 0.01).astype(int)
      ed = np.diff(np.r_[0, up, 0])
      a, b = np.where(ed == 1)[0], np.where(ed == -1)[0]
      keep = [(i, j) for i, j in zip(a, b) if j - i >= int(0.08 / dt)]
      st_n.append(len(keep))
      for i, j in keep:
        sw.append((j - i) * dt)
        lift.append(Zf[i:j, e, k].max() * 100)
        jj = min(j, len(Zf) - 1)
        length.append(np.linalg.norm(XYf[jj, e, k] - XYf[max(i - 1, 0), e, k]) * 100)
        td.append(TILT[jj, e, k])
      stance += list(TILT[up == 0, e, k][::10])
    out.append(dict(steps=np.median(st_n), swing=np.median(sw) if sw else 0, lift=np.median(lift) if lift else 0,
                    length=np.median(length) if length else 0, td=np.median(td) if td else 0,
                    stance=np.median(stance) if stance else 0))
  ds = None if F is None else np.mean((F[..., 0] > 5) & (F[..., 1] > 5)) * 100
  return out, ds


# BWC reference: the clip's first forward segment (command [1,0,0], 12.82-34.17 s).
mterm = u.command_manager.get_term("motion")
mot = mterm.motion
bfeet = [mterm.cfg.body_names.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
fps = 1.0 / dt
a, b = int(12.82 * fps), int(34.17 * fps)
bz = mot.body_pos_w[a:b, bfeet, 2].cpu().numpy()[:, None, :]
bz = bz - np.median(mot.body_pos_w[int(5 * fps):int(10 * fps), bfeet, 2].cpu().numpy(), axis=0)
bxy = mot.body_pos_w[a:b, bfeet, :2].cpu().numpy()[:, None]
btilt = tilt_deg(mot.body_quat_w[a:b, bfeet].cpu().numpy())[:, None]
bwc, _ = gait(bz, bxy, btilt)
bwc_adv = float((mot.body_pos_w[b, 0, 0] - mot.body_pos_w[a, 0, 0]).item()) * 12.0 / ((b - a) * dt)

Z, F, P, YAW, XY, TL = [], [], [], [], [], []
with torch.no_grad():
  for _ in range(int(3.0 / dt)):
    obs, _, _, _ = env.step(torch.zeros(N, env.num_actions))
  for _ in range(int(2.0 / dt)):
    obs, _, _, _ = env.step(pol(obs))
  z_stand = robot.data.body_link_pos_w[:, feet, 2].clone()
  p0 = robot.data.root_link_pos_w[:, :2].clone()
  CMD[:, 0] = 1.0
  for _ in range(int(12.0 / dt)):
    obs, _, _, _ = env.step(pol(obs))
    Z.append((robot.data.body_link_pos_w[:, feet, 2] - z_stand).numpy().copy())
    XY.append(robot.data.body_link_pos_w[:, feet, :2].numpy().copy())
    TL.append(tilt_deg(robot.data.body_link_quat_w[:, feet].numpy()))
    F.append(sens.data.force.norm(dim=-1).squeeze(-1).numpy().copy())
    P.append((robot.data.root_link_pos_w[:, :2] - p0).numpy().copy())
    q = robot.data.root_link_quat_w
    YAW.append(torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)).numpy().copy())
Z, F, P, XY, TL = np.stack(Z), np.stack(F), np.stack(P), np.stack(XY), np.stack(TL)
YAW = np.degrees(np.unwrap(np.stack(YAW), axis=0))
pol, ds = gait(Z, XY, TL, F)
print(f"{run} {ck}  retard {lag * cfg.sim.mujoco.timestep * 1000:.0f} ms, tau {TAU}  (16 departs, medianes)")
print(f"{'':22s}{'policy G':>10s}{'policy D':>10s}{'BWC G':>8s}{'BWC D':>8s}")
rows = (("pas / 12 s", "steps", "{:.0f}"), ("duree de vol (s)", "swing", "{:.2f}"), ("levee (cm)", "lift", "{:.1f}"),
        ("longueur de pas (cm)", "length", "{:.1f}"), ("pied a la pose (deg)", "td", "{:.1f}"),
        ("pied en appui (deg)", "stance", "{:.1f}"))
for name, key, f in rows:
  bv = [bwc[k][key] * (12.0 / ((b - a) * dt) if key == "steps" else 1) for k in range(2)]
  print(f"{name:22s}" + "".join(f"{f.format(pol[k][key]):>10s}" for k in range(2))
        + "".join(f"{f.format(bv[k]):>8s}" for k in range(2)))
print(f"  double appui {ds:.0f} % (BWC 28 %), avance {np.median(P[-1, :, 0]):.2f} m (BWC {bwc_adv:.2f} m),"
      f" lateral {np.median(P[-1, :, 1]):+.2f} m, lacet {np.median(YAW[-1] - YAW[0]):+.0f} deg en 12 s")
