"""BWC reference with the human upper-body swing grafted on, phase-aligned.

  uv run python scripts/tools/blend_reference.py <alpha> <out.npz> [<video.mp4>]

Stage 1 of the BWC -> 100STYLE shift (Leo, 2026-10-19): the BWC stays the base,
its contacts, root and legs untouched. For every BWC stride (left touchdown to
left touchdown) the frame's phase indexes the mean 100STYLE Neutral cycle of the
same direction (FW forward, BW backward, SW lateral; yaw keeps the BWC) and

    q_j += alpha * w(t) * (cycle_j(phase) - mean(cycle_j))     for j in UPPER

so the BWC's mean posture (clear of the QP collision zones) stays and only the
human oscillation is added. w ramps over the first and last stride of a walk.
Body poses and velocities are rebuilt by FK, in the bwc_motion_v3.npz format.
Needs ~/datasets/100style/retarget/Neutral_{FW,BW,SW}.npz (retarget_lafan cache).
"""

import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import retarget_lafan as rt  # noqa: E402

UPPER = ("CHEST_Y", "CHEST_P", "L_SHOULDER_P", "L_SHOULDER_R", "L_ELBOW_P",
         "R_SHOULDER_P", "R_SHOULDER_R", "R_ELBOW_P")
BINS = 100
CACHE = os.path.expanduser("~/datasets/100style/retarget")
CLIP_DIR = {0: ("FW", 1), 1: ("SW", 1), 2: ("BW", 1), 3: ("SW", -1)}  # BWC clip -> source, lateral sign


def touchdowns(up):
  return np.where(np.diff(up.astype(int)) == -1)[0] + 1


def mirror_cycle(c, jn):
  """Left/right mirror of a cycle (BINS, len(UPPER)) shifted half a stride."""
  from mjlab.tasks.velocity.config.rhps1.rl_ext import _joint_sign, _mirror_joint_name

  out = np.zeros_like(c)
  for k, n in enumerate(UPPER):
    out[:, UPPER.index(_mirror_joint_name(n))] = np.roll(c[:, k], BINS // 2) * _joint_sign(n)
  return out


def human_cycle(src: str, lat_sign: int, jn):
  z = np.load(f"{CACHE}/Neutral_{src}.npz")
  q, dt, sole = z["qpos"], float(z["dt"]), z["sole_z"]
  ji = [7 + jn.index(n) for n in UPPER]
  yaw = np.unwrap(Rot.from_quat(q[:, 3:7], scalar_first=True).as_euler("ZYX")[:, 0])
  td = touchdowns(sole[:, 0] > 0.01)
  cycles = []
  for a, b in zip(td[:-1], td[1:]):
    T = (b - a) * dt
    if not 0.7 < T < 1.6:
      continue
    d = q[b, :2] - q[a, :2]
    c, s = np.cos(yaw[a]), np.sin(yaw[a])
    vx, vy = (d[0] * c + d[1] * s) / T, (-d[0] * s + d[1] * c) / T
    if src == "FW" and not vx > 0.3:
      continue
    if src == "BW" and not vx < -0.2:
      continue
    if src == "SW" and not (vy * lat_sign > 0.1 and abs(vx) < 0.15):
      continue
    ph = np.linspace(0, 1, BINS + 1)
    t = np.arange(a, b + 1)
    cy = np.stack([np.interp(ph, (t - a) / (b - a), q[t, j]) for j in ji], -1)
    cycles.append(cy - ph[:, None] * (cy[-1] - cy[0]))  # periodic: no jump at touchdown
  c = np.median(np.array(cycles), 0)[:-1]
  F = np.fft.rfft(c, axis=0)
  F[4:] = 0  # first three harmonics: the swing, not the mocap noise
  c = np.fft.irfft(F, n=BINS, axis=0)
  if src != "SW":  # lateral walking is not left-right symmetric; the mirror of SW+ is SW-
    c = 0.5 * (c + mirror_cycle(c, jn))
  return c - c.mean(0), len(cycles)


def pelvis_cycle(src: str, lat_sign: int):
  """Mean 100STYLE stride of the pelvis height, zero-mean, periodic, 4 harmonics."""
  z = np.load(f"{CACHE}/Neutral_{src}.npz")
  q, dt, sole = z["qpos"], float(z["dt"]), z["sole_z"]
  yaw = np.unwrap(Rot.from_quat(q[:, 3:7], scalar_first=True).as_euler("ZYX")[:, 0])
  td = touchdowns(sole[:, 0] > 0.01)
  cycles = []
  for a, b in zip(td[:-1], td[1:]):
    T = (b - a) * dt
    if not 0.7 < T < 1.6:
      continue
    d = q[b, :2] - q[a, :2]
    c, s = np.cos(yaw[a]), np.sin(yaw[a])
    vx, vy = (d[0] * c + d[1] * s) / T, (-d[0] * s + d[1] * c) / T
    if (src == "FW" and not vx > 0.3) or (src == "BW" and not vx < -0.2) or \
       (src == "SW" and not (vy * lat_sign > 0.1 and abs(vx) < 0.15)):
      continue
    ph = np.linspace(0, 1, BINS + 1)
    t = np.arange(a, b + 1)
    cy = np.interp(ph, (t - a) / (b - a), q[t, 2])
    cycles.append(cy - ph * (cy[-1] - cy[0]))
  c = np.median(np.array(cycles), 0)[:-1]
  F = np.fft.rfft(c)
  F[0] = 0
  F[5:] = 0
  return np.fft.irfft(F, n=BINS)


def leg_ik(m, d, q, targets, leg_dofs, iters=20, limits=None):
  """Move only the leg joints of q so each ankle body meets its (pos, rot) target.
  limits=(lo, hi) over leg_dofs clamps every step."""
  d.qpos[:] = q
  for _ in range(iters):
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    J, e = [], []
    for b, (pt, Rt) in targets.items():
      jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
      mujoco.mj_jacBody(m, d, jp, jr, b)
      J += [jp[:, leg_dofs], 0.3 * jr[:, leg_dofs]]
      e += [pt - d.xpos[b], 0.3 * Rot.from_matrix(Rt @ d.xmat[b].reshape(3, 3).T).as_rotvec()]
    J, e = np.vstack(J), np.concatenate(e)
    if np.linalg.norm(e) < 1e-5:
      break
    dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(leg_dofs)), J.T @ e)
    d.qpos[np.array(leg_dofs) + 1] += dq
    if limits is not None:
      d.qpos[np.array(leg_dofs) + 1] = np.clip(d.qpos[np.array(leg_dofs) + 1], *limits)
  mujoco.mj_kinematics(m, d)
  e = np.concatenate([pt - d.xpos[b] for b, (pt, _) in targets.items()])
  return d.qpos.copy(), float(np.linalg.norm(e))


def main():
  alpha, out = float(sys.argv[1]), sys.argv[2]
  # BLEND_ALPHA_CLIP="1:0,2:0.5,3:0": per BWC clip (0 fwd, 1 left, 2 back, 3 right)
  clip_alpha = {int(k): float(v) for k, v in (kv.split(":") for kv in
                os.environ.get("BLEND_ALPHA_CLIP", "").split(",") if kv)}
  # BLEND_PELVIS=1 (stage 2a): add the human pelvis bob, legs re-solved by IK so both
  # ankles keep their BWC world pose; BLEND_PELVIS_CLIP per clip as above.
  pel = float(os.environ.get("BLEND_PELVIS", "0"))
  pel_clip = {int(k): float(v) for k, v in (kv.split(":") for kv in
              os.environ.get("BLEND_PELVIS_CLIP", "").split(",") if kv)}
  m = rt.model()
  jn = [m.joint(j).name for j in range(1, m.njnt)]
  z = dict(np.load("docs/bwc_motion_v3.npz"))
  fps = float(z["fps"][0])
  cs = list(z["clip_starts"]) + [len(z["joint_pos"])]
  iL, iR = m.body("L_ANKLE_P_LINK").id - 1, m.body("R_ANKLE_P_LINK").id - 1
  q = np.concatenate([z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]], 1)
  ju = [jn.index(n) for n in UPPER]
  for k, (a, b) in enumerate(zip(cs[:-1], cs[1:])):
    if k not in CLIP_DIR:
      continue
    src, sgn = CLIP_DIR[k]
    a_k = clip_alpha.get(k, alpha)
    cyc, n = human_cycle(src, sgn, jn)
    zl = z["body_pos_w"][a:b, iL, 2]
    base = min(zl.min(), z["body_pos_w"][a:b, iR, 2].min())
    td = touchdowns(zl > base + 0.01)
    w = np.zeros(b - a)
    ph = np.zeros(b - a)
    for s, (t0, t1) in enumerate(zip(td[:-1], td[1:])):
      ph[t0:t1] = (np.arange(t0, t1) - t0) / (t1 - t0)
      if s == 0:
        w[t0:t1] = ph[t0:t1]
      elif s == len(td) - 2:
        w[t0:t1] = 1 - ph[t0:t1]
      else:
        w[t0:t1] = 1
    x = ph * BINS
    i0 = np.floor(x).astype(int) % BINS
    fr = (x - np.floor(x))[:, None]
    val = (1 - fr) * cyc[i0] + fr * cyc[(i0 + 1) % BINS]
    q[a:b, 7 + np.array(ju)] += a_k * w[:, None] * val
    p_k = pel_clip.get(k, pel)
    if p_k > 0:
      pc = pelvis_cycle(src, sgn)
      dz = p_k * w * ((1 - fr[:, 0]) * pc[i0] + fr[:, 0] * pc[(i0 + 1) % BINS])
      dd = mujoco.MjData(m)
      leg_dofs = [m.joint(n).dofadr[0] for n in jn if any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]
      bl, br_ = m.body("L_ANKLE_P_LINK").id, m.body("R_ANKLE_P_LINK").id
      worst = 0.0
      for i in np.where(np.abs(dz) > 1e-5)[0]:
        x = q[a + i].copy()
        dd.qpos[:] = x
        mujoco.mj_kinematics(m, dd)
        tg = {b_: (dd.xpos[b_].copy(), dd.xmat[b_].reshape(3, 3).copy()) for b_ in (bl, br_)}
        x[2] += dz[i]
        x, err = leg_ik(m, dd, x, tg, leg_dofs)
        q[a + i] = x
        worst = max(worst, err)
      print(f"clip {k}: bassin +/- {np.ptp(dz) * 50:.1f} cm (x{p_k:g}), residu IK des chevilles max {worst * 1000:.1f} mm")
    amp = np.degrees(np.ptp(cyc, 0))
    print(f"clip {k} ({src}{'+' if sgn > 0 else '-'}, {n} foulees humaines): ajout crete a crete "
          + ", ".join(f"{u} {a_k * x:.0f}" for u, x in zip(UPPER, amp)) + " deg")
  # limits, then FK and velocities
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  q[:, 7:] = np.clip(q[:, 7:], lo, hi)
  d = mujoco.MjData(m)
  pos = np.zeros_like(z["body_pos_w"])
  quat = np.zeros_like(z["body_quat_w"])
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(m, d)
    pos[i], quat[i] = d.xpos[1:], d.xquat[1:]
  dt = 1.0 / fps
  out_d = dict(z)
  out_d["joint_pos"] = q[:, 7:].astype(z["joint_pos"].dtype)
  jv = np.zeros_like(out_d["joint_pos"])
  lv = np.zeros_like(pos)
  av = np.zeros_like(pos)
  for a, b in zip(cs[:-1], cs[1:]):
    jv[a:b] = np.gradient(q[a:b, 7:], dt, axis=0)
    lv[a:b] = np.gradient(pos[a:b], dt, axis=0)
    R = Rot.from_quat(quat[a:b].reshape(-1, 4), scalar_first=True).as_matrix().reshape(b - a, -1, 3, 3)
    dR = np.einsum("tbij,tbkj->tbik", np.gradient(R, dt, axis=0), R)  # dR R^T = [w]x
    av[a:b] = np.stack([dR[..., 2, 1], dR[..., 0, 2], dR[..., 1, 0]], -1)
  out_d.update(joint_vel=jv, body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lv, body_ang_vel_w=av)
  out_d = {k: v.astype(z[k].dtype) for k, v in out_d.items()}
  np.savez_compressed(out, **out_d)
  print("ecrit", out)
  if len(sys.argv) > 3:
    import mediapy

    fr = []
    for k, t0 in ((0, 14.0), (1, 16.0), (2, 14.0)):
      a = int(cs[k] + t0 * fps)
      sl = slice(a, a + int(6 * fps), int(fps / 30))
      bq = np.concatenate([z["body_pos_w"][sl, 0], z["body_quat_w"][sl, 0], z["joint_pos"][sl]], 1)
      fr.append(np.concatenate([rt.panel(m, bq, "BWC actuel"),
                                rt.panel(m, q[sl], f"BWC + balancement humain x{alpha:g}")], 2))
    mediapy.write_video(sys.argv[3], np.concatenate(fr, 0), fps=30)


if __name__ == "__main__":
  main()
