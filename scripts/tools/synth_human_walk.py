"""Straight human walking for RHPS1: 100STYLE stride style, LIPM centre of mass.

  uv run python scripts/tools/synth_human_walk.py <out.npz> [<leg_amp>] [<slow>]

The mean 100STYLE Neutral stride (FW forward, BW backward; median over ~80 strides,
6 harmonics) gives the style: upper-body joints, pelvis roll/pitch/yaw and bob,
contact schedule, step length, ankle lift and swing foot orientation, replayed at the
human cadence slowed `slow` times (default 2).

Slowed, the human kinematics are not balanced (LIPM ZMP outside the sole 57% of the
time), so the centre of mass comes from the footholds, as in the BWC: the ZMP rolls
heel to toe on the stance sole, moves linearly across double support, and the CoM
solves c - zc/g c'' = zmp exactly (boundary value problem, standing at both ends).
The pelvis is moved until the whole-body CoM follows it; its height keeps both knees
>= 15 deg. Stance feet are flat where they landed, at least the BWC stance width
apart; swing feet lift at the human ankle height, travel minimum-jerk between
footholds and take the human foot orientation mid-swing; soles stay above ground.
Legs by IK on both ankles.

Gait initiation and termination: the stride amplitude ramps from the BWC's standing
pose over 1.5 strides, and back. Clips: [human forward] [BWC left] [human backward]
[BWC right] [BWC yaw left] [BWC yaw right]; bwc_motion_v3.npz format.
"""

import sys

import mujoco
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import blend_global as bg  # noqa: E402
import blend_reference as br  # noqa: E402
import retarget_lafan as rt  # noqa: E402

FPS = 200.0
G = 9.81
KNEE_MIN = np.radians(15.0)
ROLL = 0.025  # ZMP heel-to-toe travel on the stance sole, +- m
# stance = sole down and ankle within this of its lowest: double support ~25-30% like
# the BWC (full human contact gives 46-57%, the swing then whips the knee)
STANCE_H = {"FW": 0.03, "BW": 0.025}
M = rt.model()
JN = [M.joint(j).name for j in range(1, M.njnt)]
LEGS = [i for i, n in enumerate(JN) if any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]
ARMS = [i for i, n in enumerate(JN) if any(t in n for t in ("SHOULDER", "ELBOW", "WRIST"))]
ANK = {s: M.body(f"{s}_ANKLE_P_LINK").id for s in ("L", "R")}
SOLE = {s: [k for k in range(M.ngeom) if M.geom_bodyid[k] == ANK[s]
            and M.geom_type[k] == mujoco.mjtGeom.mjGEOM_BOX][0] for s in ("L", "R")}
LEG_DOFS = [M.joint(JN[j]).dofadr[0] for j in LEGS]
LEG_Q = np.array(LEG_DOFS) + 1
LEG_LIM = (np.array([M.jnt_range[M.dof_jntid[j], 0] for j in LEG_DOFS]),
           np.array([M.jnt_range[M.dof_jntid[j], 1] for j in LEG_DOFS]))


def human_stride(src):
  z = np.load(f"{br.CACHE}/Neutral_{src}.npz")
  q, dt, sole = z["qpos"], float(z["dt"]), z["sole_z"]
  pel, yaw = bg.pelvis_signals(q)
  keep = bg.direction_ok(src, 1, q, yaw, dt)
  td, lo = bg.foot_events(sole[:, 0] > 0.01)
  d = mujoco.MjData(M)
  az = np.zeros((len(q), 2))
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(M, d)
    az[i] = d.xpos[[ANK["L"], ANK["R"]], 2]
  down = (az - az.min(0) < STANCE_H[src]) & (sole < 0.01)
  sig = np.concatenate([q[:, 7:], pel[:, 1:], down.astype(float), az - az.min(0), pel[:, :1]], 1)
  cyc, _ = bg.stride_cycles(sig, td, lo, keep)
  periods = [(b - a) * dt for a, b in zip(td[:-1], td[1:], strict=True) if keep(a, b)]
  c = np.median(cyc, 0)
  joints = bg.smooth_cycle(c[:, :30])
  rot = bg.smooth_cycle(c[:, 30:33] - c[:, 30:33].mean(0))
  stance = c[:, 33:35] > 0.5
  lift = np.maximum(bg.smooth_cycle(c[:, 35:37]) - 0.012, 0.0)  # ankle height above flat-down
  bob = bg.smooth_cycle(c[:, 37:38] - c[:, 37].mean())[:, 0]
  return joints, rot, stance, lift, bob, float(np.median(periods)), len(cyc)


def bwc_stand():
  z = np.load("docs/bwc_motion_v3.npz")
  i = int(5 * FPS)  # well inside the standing prologue of clip 0
  q = np.concatenate([z["body_pos_w"][i, 0], z["body_quat_w"][i, 0], z["joint_pos"][i]])
  return q.astype(float)


def smooth(x):
  x = float(np.clip(x, 0, 1))
  return x * x * (3 - 2 * x)


def lipm_com(p, zc):
  """CoM (T, k) whose LIPM ZMP is exactly p: c - zc/g c'' = p, c = p at both ends."""
  n, k = len(p), zc / G * FPS ** 2
  ab = np.zeros((3, n))
  ab[0, 1:] = -k
  ab[1] = 1 + 2 * k
  ab[2, :-1] = -k
  ab[1, 0] = ab[1, -1] = 1.0
  ab[0, 1] = ab[2, -2] = 0.0
  return solve_banded((1, 1), ab, p)


def synth(src, walk_s, leg_amp, slow, stand_s=4.0, ramp_strides=1.5):
  joints, rot, stance, lift, bob, period, n = human_stride(src)
  T = period * slow
  qs = bwc_stand()
  q_stand = qs[7:]
  mean = joints.mean(0)
  cyc = joints.copy()
  cyc[:, LEGS] = mean[LEGS] + leg_amp * (joints[:, LEGS] - mean[LEGS])
  cyc[:, ARMS] = q_stand[ARMS] + (joints[:, ARMS] - mean[ARMS])
  ramp = ramp_strides * T
  total = stand_s + ramp + walk_s + ramp + stand_s
  t = np.arange(0, total, 1 / FPS)
  n_t = len(t)
  a = np.clip((t - stand_s) / ramp, 0, 1) * np.clip((total - stand_s - t) / ramp, 0, 1)
  a = a * a * (3 - 2 * a)  # smoothstep
  ph = np.where(t > stand_s, (t - stand_s) / T, 0.0) % 1.0
  cj = bg.lookup(cyc, ph)
  cr = bg.lookup(rot, ph)
  st = bg.lookup(stance.astype(float), ph) > 0.5
  cl = bg.lookup(lift, ph) * a[:, None]
  cb = bg.lookup(bob[:, None], ph)[:, 0] * a
  st[a < 0.15] = True  # small steps near the stand: both feet down
  qj = q_stand[None] + a[:, None] * (cj - q_stand[None])
  R_stand = Rot.from_quat(qs[3:7], scalar_first=True)
  R0 = Rot.from_euler("Z", R_stand.as_euler("ZYX")[0])
  fwd, lat = R0.apply([1.0, 0, 0])[:2], R0.apply([0, 1.0, 0])[:2]
  Rp = [R0 * Rot.from_rotvec(a[i] * cr[i]) * (R0.inv() * R_stand) for i in range(n_t)]

  # stand geometry: ankles, soles, CoM, hips, leg reach at KNEE_MIN
  d = mujoco.MjData(M)
  d.qpos[:] = qs
  mujoco.mj_kinematics(M, d)
  mujoco.mj_comPos(M, d)
  ground = np.array([d.xpos[ANK[s], 2] for s in ("L", "R")])
  flat = {s: d.xmat[ANK[s]].reshape(3, 3).copy() for s in ("L", "R")}
  sole_off = {s: (d.geom_xpos[SOLE[s]] - d.xpos[ANK[s]])[:2] for s in ("L", "R")}
  sole0 = rt.sole_min_z(M, d)
  com0 = d.subtree_com[1].copy()
  zc = com0[2] - np.mean(sole0)
  mid0 = 0.5 * (d.xpos[ANK["L"], :2] + d.xpos[ANK["R"], :2])
  w0 = (d.xpos[ANK["L"], :2] - d.xpos[ANK["R"], :2]) @ lat
  hip_off, reach = {}, {}
  for s in ("L", "R"):
    h, k_, an = (d.xanchor[M.joint(f"{s}_{j}").id].copy() for j in ("CROTCH_P", "KNEE_P", "ANKLE_P"))
    hip_off[s] = R_stand.inv().apply(h - qs[:3])
    lt, ls = np.linalg.norm(h - k_), np.linalg.norm(k_ - an)
    reach[s] = np.sqrt(lt ** 2 + ls ** 2 + 2 * lt * ls * np.cos(KNEE_MIN))

  # cycle kinematics with the pelvis at the origin: ankle positions and orientations
  rel = np.zeros((n_t, 2, 3))
  Rc = np.zeros((n_t, 2, 3, 3))
  qk = np.tile(qs, (n_t, 1))
  for i in range(n_t):
    qk[i, :3] = 0.0
    qk[i, 3:7] = Rp[i].as_quat(scalar_first=True)
    qk[i, 7:] = qj[i]
    d.qpos[:] = qk[i]
    mujoco.mj_kinematics(M, d)
    rel[i] = [d.xpos[ANK["L"]], d.xpos[ANK["R"]]]
    Rc[i] = [d.xmat[ANK["L"]].reshape(3, 3), d.xmat[ANK["R"]].reshape(3, 3)]
  # footholds from the cycle's own progression: the pelvis moves by -dr of the feet down
  pc = np.zeros((n_t, 2))
  pc[0] = qs[:2]
  for i in range(1, n_t):
    on = st[i] & st[i - 1]
    pc[i] = pc[i - 1] - ((rel[i] - rel[i - 1])[on, :2].mean(0) if on.any() else 0.0)

  def footholds(widen):
    hold = np.zeros((n_t, 2, 3))
    for si, sgn in ((0, 1.0), (1, -1.0)):
      cur = np.r_[qs[:2] + rel[0, si, :2], ground[si]]
      for i in range(n_t):
        if st[i, si] and (i == 0 or not st[i - 1, si]):
          cur = np.r_[pc[i] + rel[i, si, :2] + sgn * 0.5 * widen * a[i] * lat, ground[si]]
        hold[i, si] = cur
    return hold

  hold = footholds(0.0)
  ds = st.all(1) & (a > 0.9)
  w_h = float(np.median((hold[ds, 0, :2] - hold[ds, 1, :2]) @ lat))
  if w_h < w0:  # the human walks narrower than the robot's legs allow
    hold = footholds(w0 - w_h)

  # foot targets (pelvis independent)
  tpos = hold.copy()
  trot = np.zeros((n_t, 2, 3, 3))
  for si, s in enumerate(("L", "R")):
    i = 0
    while i < n_t:
      if st[i, si]:
        trot[i, si] = flat[s]
        i += 1
        continue
      i0 = i
      while i < n_t and not st[i, si]:
        i += 1
      i1 = i - 1
      p0, p1 = hold[max(i0 - 1, 0), si], hold[min(i1 + 1, n_t - 1), si]
      for j in range(i0, i1 + 1):
        u = (j - i0 + 0.5) / (i1 - i0 + 1)
        sx = np.clip((u - 0.05) / 0.9, 0, 1)
        tpos[j, si] = p0 + sx ** 3 * (10 - 15 * sx + 6 * sx * sx) * (p1 - p0)
        tpos[j, si, 2] = ground[si] + cl[j, si] * smooth(min(u, 1 - u) / 0.15)
        w = 1 - smooth(min(u, 1 - u) / 0.35)
        rv = Rot.from_matrix(flat[s] @ Rc[j, si].T).as_rotvec() * w
        trot[j, si] = Rot.from_rotvec(rv).as_matrix() @ Rc[j, si]

  # ZMP: rolls on the stance sole in single support, linear across double support,
  # under the standing CoM while both feet are down near the stand
  roll = 1.0 if src == "FW" else -1.0
  zmp = np.full((n_t, 2), np.nan)
  ss = st.sum(1) == 1
  i = 0
  while i < n_t:
    if not ss[i]:
      i += 1
      continue
    si = int(np.argmax(st[i]))
    i0 = i
    while i < n_t and ss[i] and st[i, si]:
      i += 1
    u = (np.arange(i0, i) - i0 + 0.5) / (i - i0)
    zmp[i0:i] = hold[i0, si, :2] + sole_off["LR"[si]] + (roll * ROLL * a[i0:i] * (2 * u - 1))[:, None] * fwd
  still = st.all(1) & (a < 0.15)
  zmp[still] = 0.5 * (hold[still, 0, :2] + hold[still, 1, :2]) + (com0[:2] - mid0)
  for c in range(2):
    ok = ~np.isnan(zmp[:, c])
    zmp[:, c] = np.interp(np.arange(n_t), np.where(ok)[0], zmp[ok, c])
  com_ref = lipm_com(zmp, zc)

  def pelvis_z(pxy):
    z_nom = qs[2] + cb
    bound = np.full(n_t, np.inf)
    for i in range(n_t):
      for si, s in enumerate(("L", "R")):
        h = Rp[i].apply(hip_off[s])
        dxy = np.linalg.norm(pxy[i] + h[:2] - tpos[i, si, :2])
        bound[i] = min(bound[i], tpos[i, si, 2] - h[2] + np.sqrt(max(reach[s] ** 2 - dxy ** 2, 0.0)))
    k = int(0.4 * FPS)
    return uniform_filter1d(minimum_filter1d(np.minimum(z_nom, bound), k, mode="nearest"), k, mode="nearest")

  def solve(pxy, pz, every=1, final=False):
    idx = np.r_[np.arange(0, n_t, every), n_t - 1] if (n_t - 1) % every else np.arange(0, n_t, every)
    q = np.zeros((n_t, 37))
    com = np.zeros((len(idx), 2))
    prev, worst = None, 0.0
    for j, i in enumerate(idx):
      x = qk[i].copy()
      x[:2], x[2] = pxy[i], pz[i]
      if prev is not None:
        x[LEG_Q] = prev[LEG_Q]  # warm start
      tg = {ANK[s]: (tpos[i, si].copy(), trot[i, si]) for si, s in enumerate(("L", "R"))}
      x, err = br.leg_ik(M, d, x, tg, LEG_DOFS, iters=25, limits=LEG_LIM)
      for _ in range(3 if final else 0):  # a pitched swing foot near the ground: raise it
        sz = rt.sole_min_z(M, d)
        low = {si: sole0[si] - sz[si] for si in range(2) if not st[i, si] and sz[si] < sole0[si] - 1e-4}
        if not low:
          break
        for si, dz in low.items():
          b = ANK["LR"[si]]
          tg[b] = (tg[b][0] + np.array([0.0, 0.0, dz]), tg[b][1])
        x, err = br.leg_ik(M, d, x, tg, LEG_DOFS, iters=25, limits=LEG_LIM)
      mujoco.mj_comPos(M, d)
      com[j] = d.subtree_com[1][:2]
      worst = max(worst, err)
      prev = q[i] = x
    full = np.stack([np.interp(np.arange(n_t), idx, com[:, c]) for c in range(2)], 1)
    return q, full, worst

  # pelvis xy until the whole-body CoM follows com_ref
  pxy = com_ref + (qs[:2] - com0[:2])
  for _ in range(3):
    pz = pelvis_z(pxy)
    _, com, _ = solve(pxy, pz, every=4)
    pxy = pxy + (com_ref - com)
  pz = pelvis_z(pxy)
  q, com, worst = solve(pxy, pz, final=True)
  kn = np.degrees(q[:, 7 + JN.index("L_KNEE_P")])
  walk = a > 0.9
  print(f"  {src}: appui double {st[walk].all(1).mean() * 100:.0f}%, largeur humaine {w_h * 100:.0f} cm (BWC {w0 * 100:.0f}),"
        f" CoM - LIPM max {np.abs(com - com_ref).max() * 1000:.1f} mm, genou {kn.min():.0f}-{kn.max():.0f} deg,"
        f" residu IK des pieds max {worst * 1000:.1f} mm")
  return q, T, n


def frames(q, fps):
  d = mujoco.MjData(M)
  pos = np.zeros((len(q), M.nbody - 1, 3))
  quat = np.zeros((len(q), M.nbody - 1, 4))
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(M, d)
    pos[i], quat[i] = d.xpos[1:], d.xquat[1:]
  dt = 1 / fps
  jv = np.gradient(q[:, 7:], dt, axis=0)
  lv = np.gradient(pos, dt, axis=0)
  Rm = Rot.from_quat(quat.reshape(-1, 4), scalar_first=True).as_matrix().reshape(len(q), -1, 3, 3)
  dR = np.einsum("tbij,tbkj->tbik", np.gradient(Rm, dt, axis=0), Rm)
  av = np.stack([dR[..., 2, 1], dR[..., 0, 2], dR[..., 1, 0]], -1)
  return q[:, 7:], jv, pos, quat, lv, av


def main():
  out = sys.argv[1]
  leg_amp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
  slow = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
  z = dict(np.load("docs/bwc_motion_v3.npz"))
  cs = list(z["clip_starts"]) + [len(z["joint_pos"])]
  parts = []
  for k in range(6):
    if k in (0, 2):
      src = "FW" if k == 0 else "BW"
      q, T, n = synth(src, 16.0, leg_amp, slow)
      jp, jv, pos, quat, lv, av = frames(q, FPS)
      print(f"clip {k} humain {src} : {n} foulees, foulee {T:.2f} s, {len(q) / FPS:.0f} s")
      parts.append(dict(joint_pos=jp, joint_vel=jv, body_pos_w=pos, body_quat_w=quat,
                        body_lin_vel_w=lv, body_ang_vel_w=av))
    else:
      a, b = cs[k], cs[k + 1]
      parts.append({key: z[key][a:b] for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                                                  "body_lin_vel_w", "body_ang_vel_w")})
  o = {key: np.concatenate([p[key] for p in parts]).astype(z[key].dtype) for key in parts[0]}
  o["fps"] = z["fps"]
  o["clip_starts"] = np.cumsum([0] + [len(p["joint_pos"]) for p in parts[:-1]]).astype(z["clip_starts"].dtype)
  np.savez_compressed(out, **o)
  print("ecrit", out, "clips", o["clip_starts"])


if __name__ == "__main__":
  main()
