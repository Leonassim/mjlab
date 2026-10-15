"""Dynamic signature of a gait: the BWC reference clips against a policy in mjlab.

  uv run python scripts/tools/gait_dynamics.py <run> <ckpt> [out.npz]

Same quantities on both, through one CPU MuJoCo model of the RHPS1 so the
reference (kinematic) and the policy (simulated) are measured identically:
centroidal angular momentum (about the CoM), CoM oscillation and horizontal
acceleration, head angular velocity, trunk tilt, arm swing, knee flexion. The
policy adds what the clip cannot give: foot impact peak and loading rate, leg
torques and mechanical power. Walking part only, from 3 s after the command.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.RHPS1.rhps1_constants import get_rhps1_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
MOTION = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
CMDS = {"avant": (1, 0, 0), "gauche": (0, 1, 0), "arriere": (-1, 0, 0), "droite": (0, -1, 0),
        "rot+": (0, 0, 1), "rot-": (0, 0, -1)}
PER = 4
G = 9.81


# ---------------------------------------------------------------- CPU model
_ent = Entity(get_rhps1_robot_cfg())
M = _ent.spec.compile()
D = mujoco.MjData(M)
MASS = float(M.body_subtreemass[1])
FREE_Q, FREE_V = M.jnt_qposadr[0], M.jnt_dofadr[0]
assert M.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def _jid(name):
  for i in range(M.njnt):
    n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, i)
    if n == name or (n and n.endswith("/" + name)):
      return i
  raise KeyError(name)


def _bid(name):
  for i in range(M.nbody):
    n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, i)
    if n == name or (n and n.endswith("/" + name)):
      return i
  raise KeyError(name)


ROOT, HEAD, CHEST = _bid("BODY"), _bid("HEAD_P_LINK"), _bid("CHEST_P_LINK")


def _gid(name):
  for i in range(M.ngeom):
    n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_GEOM, i)
    if n and (n == name or n.endswith("/" + name)):
      return i
  return None


# Arm against body: the pairs the QP and the proximity terms watch.
_ARM = [f"rhps1_collision_{s}_{l}_LINK" for s in "LR" for l in ("ELBOW_Y", "WRIST_Y", "SHOULDER_Y")]
_BODY = ["rhps1_collision_CHEST_P_LINK", "rhps1_collision_BODY"] + [
  f"rhps1_collision_{s}_CROTCH_P_LINK" for s in "LR"]
ARM_PAIRS = [(_gid(a), _gid(b)) for a in _ARM for b in _BODY
             if _gid(a) is not None and _gid(b) is not None
             and not ("SHOULDER" in a and "CROTCH" in b)]


def _quat_rotate_inv(q, v):
  """World vector v into the frame of quaternion q (w, x, y, z)."""
  w, x, y, z = q
  r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
  return r.T @ v


def kinematics(root_pos, root_quat, root_linvel_w, root_angvel_w, qj, dqj, joint_names):
  """CoM, centroidal angular momentum, head angular velocity, chest tilt per frame."""
  qadr = [M.jnt_qposadr[_jid(n)] for n in joint_names]
  vadr = [M.jnt_dofadr[_jid(n)] for n in joint_names]
  T = len(root_pos)
  com, L, head_w, chest_rp = np.zeros((T, 3)), np.zeros((T, 3)), np.zeros((T, 3)), np.zeros((T, 2))
  pd = np.full(((T + 3) // 4, len(ARM_PAIRS)), np.nan)
  ft = np.zeros(6)
  vel6 = np.zeros(6)
  for t in range(T):
    D.qpos[FREE_Q:FREE_Q + 3] = root_pos[t]
    D.qpos[FREE_Q + 3:FREE_Q + 7] = root_quat[t]
    D.qpos[qadr] = qj[t]
    D.qvel[FREE_V:FREE_V + 3] = root_linvel_w[t]
    D.qvel[FREE_V + 3:FREE_V + 6] = _quat_rotate_inv(root_quat[t], root_angvel_w[t])
    D.qvel[vadr] = dqj[t]
    mujoco.mj_forward(M, D)
    mujoco.mj_subtreeVel(M, D)
    com[t] = D.subtree_com[ROOT]
    L[t] = D.subtree_angmom[ROOT]
    mujoco.mj_objectVelocity(M, D, mujoco.mjtObj.mjOBJ_BODY, HEAD, vel6, 0)
    head_w[t] = vel6[:3]
    xq = D.xquat[CHEST]
    w, x, y, z = xq
    chest_rp[t] = [np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                   np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))]
    if t % 4 == 0:
      pd[t // 4] = [mujoco.mj_geomDistance(M, D, a, b, 0.2, ft) for a, b in ARM_PAIRS]
  # Mesh-mesh distance has two artefacts: an exact 0 with no approach, and a
  # one-sample sign flip (6.24, -6.23, 6.22 cm). Both are dropped.
  pd[pd == 0.0] = np.nan
  flip = (pd[1:-1] < 0) & (np.abs(pd[1:-1] + pd[:-2]) < 0.01) & (np.abs(pd[1:-1] + pd[2:]) < 0.01)
  pd[1:-1][flip] = np.nan
  ok = ~np.isnan(pd).all(1)
  kinematics.clearance = np.nanmin(pd[ok], axis=1)
  return com, L, head_w, chest_rp


def _smooth(x, n=5):
  """Moving average with edge padding (zero padding faked huge edge accelerations)."""
  k = np.ones(n) / n
  pad = n // 2
  return np.apply_along_axis(
    lambda c: np.convolve(np.pad(c, pad, mode="edge"), k, mode="valid"), 0, x)


def metrics(dt, com, L, head_w, chest_rp, qj, joint_names, heading, forces=None, tau=None, dqj=None):
  out = {}
  out["L_lacet"] = np.sqrt(np.mean(L[:, 2] ** 2))
  out["L_roulis_tangage"] = np.sqrt(np.mean(L[:, 0] ** 2 + L[:, 1] ** 2))
  z = com[:, 2] - com[:, 2].mean()
  out["CdM_vertical_cm"] = (np.percentile(z, 95) - np.percentile(z, 5)) * 100
  t = np.arange(len(com))
  lat_axis = np.array([-np.sin(heading), np.cos(heading)])
  xy = com[:, :2] - np.stack([np.polyval(np.polyfit(t, com[:, i], 1), t) for i in range(2)], 1)
  lat = xy @ lat_axis
  out["CdM_lateral_cm"] = (np.percentile(lat, 95) - np.percentile(lat, 5)) * 100
  v = np.gradient(_smooth(com[:, :2], 9), dt, axis=0)
  a = np.gradient(_smooth(v, 9), dt, axis=0)
  out["CdM_acc_horiz"] = np.sqrt(np.mean((a ** 2).sum(1)))
  c = getattr(kinematics, "clearance", None)
  if c is not None and len(c):
    out["bras_corps_min_cm"] = float(np.min(c) * 100)
    out["bras_corps_p5_cm"] = float(np.percentile(c, 5) * 100)
  out["tete_degps"] = np.degrees(np.sqrt(np.mean((head_w ** 2).sum(1))))
  out["tronc_roulis_deg"] = np.degrees(np.std(chest_rp[:, 0]))
  out["tronc_tangage_deg"] = np.degrees(np.std(chest_rp[:, 1]))
  ji = {n: i for i, n in enumerate(joint_names)}
  sh = [qj[:, ji[s + "_SHOULDER_P"]] for s in "LR"]
  out["bras_deg"] = np.degrees(np.mean([np.percentile(s, 95) - np.percentile(s, 5) for s in sh]))
  kn = [qj[:, ji[s + "_KNEE_P"]] for s in "LR"]
  out["genou_ampl_deg"] = np.degrees(np.mean([np.percentile(k, 95) - np.percentile(k, 5) for k in kn]))
  if forces is not None:
    peaks, rates = [], []
    for j in range(forces.shape[1]):
      f = forces[:, j]
      on = np.where((f[1:] > 20) & (f[:-1] <= 20))[0] + 1
      for o in on:
        w = f[o:o + int(0.1 / dt)]
        if len(w) < int(0.1 / dt):
          continue
        peaks.append(w.max() / (MASS * G))
        k = min(int(0.05 / dt), len(w) - 1)
        rates.append(w[k] / (MASS * G) / (k * dt))
    out["impact_pic_PC"] = np.median(peaks) if peaks else np.nan
    out["mise_en_charge_PC_s"] = np.median(rates) if rates else np.nan
  if tau is not None:
    legs = [i for i, n in enumerate(joint_names) if any(s in n for s in ("CROTCH", "KNEE", "ANKLE"))]
    out["couple_jambes_rms"] = np.sqrt(np.mean(tau[:, legs] ** 2))
    out["puissance_W"] = np.mean(np.abs(tau * dqj).sum(1))
  return out


def reference(names):
  """Metrics of each v3 clip, walking part only (first lift + 2 s to last lift - 1 s)."""
  m = np.load(MOTION)
  fps = float(m["fps"][0])
  starts = list(m["clip_starts"]) + [len(m["joint_pos"])]
  bi = [_ent_names.index(n) for n in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
  res = {}
  for name, a, b in zip(["avant", "gauche", "arriere", "droite", "rot+", "rot-"], starts[:-1], starts[1:]):
    zz = m["body_pos_w"][a:b][:, bi, 2]
    lifted = np.where((zz > zz.min(0) + 0.01).any(1))[0]
    s, e = a + lifted[0] + int(2 * fps), a + lifted[-1] - int(1 * fps)
    sl = slice(s, e)
    rq = m["body_quat_w"][sl, 0]
    yaw = np.arctan2(2 * (rq[:, 0] * rq[:, 3] + rq[:, 1] * rq[:, 2]), 1 - 2 * (rq[:, 2] ** 2 + rq[:, 3] ** 2))
    k = kinematics(m["body_pos_w"][sl, 0], rq, m["body_lin_vel_w"][sl, 0], m["body_ang_vel_w"][sl, 0],
                   m["joint_pos"][sl], m["joint_vel"][sl], names)
    res[name] = metrics(1 / fps, *k, m["joint_pos"][sl], names, float(np.mean(yaw)))
  return res


def rollout(run, ck):
  cfg = load_env_cfg(TASK, play=True)
  N = PER * len(CMDS)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = MOTION
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
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
  runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True},
              strict=True, map_location="cpu")
  pol = runner.get_inference_policy(device="cpu")
  robot = u.scene["robot"]
  sens = u.scene["feet_ground_contact"]
  dt = u.step_dt
  rec = {k: [] for k in ("pos", "quat", "lv", "av", "q", "dq", "f", "tau", "coll")}
  coll = u.scene["self_collision"] if "self_collision" in u.scene.sensors else None
  obs, _ = env.reset()
  with torch.no_grad():
    for k in range(int(20.0 / dt)):
      cmd[:] = target if k * dt >= 3.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
      if k * dt < 6.0:
        continue
      d = robot.data
      rec["pos"].append(d.root_link_pos_w.numpy() - u.scene.env_origins.numpy())
      rec["quat"].append(d.root_link_quat_w.numpy().copy())
      rec["lv"].append(d.root_link_lin_vel_w.numpy().copy())
      rec["av"].append(d.root_link_ang_vel_w.numpy().copy())
      rec["q"].append(d.joint_pos.numpy().copy())
      rec["dq"].append(d.joint_vel.numpy().copy())
      rec["f"].append(sens.data.force.norm(dim=-1).reshape(N, -1).numpy().copy())
      rec["tau"].append(d.qfrc_actuator.numpy().copy())
      rec["coll"].append(coll.data.found.reshape(N, -1).sum(-1).numpy().copy() if coll is not None
                         else np.zeros(N))
  rec = {k: np.stack(v, 1) for k, v in rec.items()}
  return rec, dt, list(robot.joint_names)


_ent_names = list(mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, i).split("/")[-1] for i in range(1, M.nbody))

if __name__ == "__main__":
  run, ck = sys.argv[1], sys.argv[2]
  rec, dt, names = rollout(run, ck)
  if len(sys.argv) > 3:
    np.savez_compressed(sys.argv[3], dt=dt, names=np.array(names), **rec)
  tau = rec["tau"]
  if tau.shape[-1] != len(names):
    tau = tau[..., -len(names):]
  pol = {}
  for i, name in enumerate(CMDS):
    per = []
    for e in range(i * PER, (i + 1) * PER):
      qq = rec["quat"][e]
      yaw = np.arctan2(2 * (qq[:, 0] * qq[:, 3] + qq[:, 1] * qq[:, 2]), 1 - 2 * (qq[:, 2] ** 2 + qq[:, 3] ** 2))
      k = kinematics(rec["pos"][e], qq, rec["lv"][e], rec["av"][e], rec["q"][e], rec["dq"][e], names)
      per.append(metrics(dt, *k, rec["q"][e], names, float(np.mean(yaw)),
                         forces=rec["f"][e], tau=tau[e], dqj=rec["dq"][e]))
    pol[name] = {k: float(np.median([p[k] for p in per])) for k in per[0]}
  ref = reference(names)
  keys = list(pol["avant"].keys())
  print(f"{run} {ck}  masse {MASS:.1f} kg, retard 30 ms  (policy | BWC reference)")
  for name in CMDS:
    print(f"== {name}")
    for k in keys:
      r = ref[name].get(k, np.nan)
      print(f"  {k:22s} {pol[name][k]:8.3f}   BWC {r:8.3f}")
