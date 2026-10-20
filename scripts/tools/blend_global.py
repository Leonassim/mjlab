"""The whole 100STYLE Neutral oscillation on the BWC at once (Leo, 2026-10-20).

  uv run python scripts/tools/blend_global.py <alpha> <out.npz> [<raise_m>]

The BWC keeps its contacts (touchdown and lift-off instants, footholds) and its
mean posture; its OSCILLATION is replaced by the human one, joint by joint:

    q(t) = q_BWC(t) + alpha * w(t) * [ (c_h - mean c_h) - (c_b - mean c_b) ](phase)

with c_h the mean 100STYLE stride (FW forward, BW backward, SW sideways; yaw keeps
the BWC) and c_b the BWC's own mean stride. Leg joints use their own foot's phase,
time-warped so the human swing falls in the BWC swing; pelvis and upper body use
the left foot's. The pelvis also gets the human height and rotation oscillation,
plus raise_m (default 0.015 m) so the stance knee can straighten. Then each leg is
re-solved by IK: a stance foot stays on its BWC pose, a swing foot keeps the human
shape mid-swing and lands on the BWC foothold (cos^2 blend toward the BWC pose at
both ends of the swing), never below the BWC sole height.
"""

import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import blend_reference as br  # noqa: E402
import retarget_lafan as rt  # noqa: E402

BINS = 100
HARM = 6
CLIP_DIR = {0: ("FW", 1), 1: ("SW", 1), 2: ("BW", 1), 3: ("SW", -1)}
LEG = ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE_P", "ANKLE_R", "ANKLE_P")


def smooth_cycle(c):
  F = np.fft.rfft(c, axis=0)
  F[HARM + 1:] = 0
  return np.fft.irfft(F, n=BINS, axis=0)


def foot_events(up):
  td = br.touchdowns(up)
  lo = np.where(np.diff(up.astype(int)) == 1)[0] + 1
  return td, lo


def stride_cycles(sig, td, lo, keep):
  """Per stride of one foot (touchdown to touchdown): resample sig (T, k) on BINS phase
  points; returns (cycles (n, BINS, k), stance fractions)."""
  cyc, frac = [], []
  for a, b in zip(td[:-1], td[1:]):
    if not keep(a, b):
      continue
    l = lo[(lo > a) & (lo < b)]
    if not len(l):
      continue
    ph = np.linspace(0, 1, BINS + 1)
    t = np.arange(a, b + 1)
    cy = np.stack([np.interp(ph, (t - a) / (b - a), sig[t, j]) for j in range(sig.shape[1])], -1)
    cyc.append((cy - ph[:, None] * (cy[-1] - cy[0]))[:-1])
    frac.append((l[0] - a) / (b - a))
  return np.array(cyc), float(np.median(frac))


def pelvis_signals(q):
  """Pelvis height and rotation vector relative to its 1 s heading frame, (T, 4)."""
  R = Rot.from_quat(q[:, 3:7], scalar_first=True)
  yaw = np.unwrap(R.as_euler("ZYX")[:, 0])
  k = 61
  ys = np.convolve(np.pad(yaw, k // 2, mode="edge"), np.ones(k) / k, "valid")
  rel = (Rot.from_euler("Z", ys).inv() * R).as_rotvec()
  return np.concatenate([q[:, 2:3], rel], 1), ys


def warp(phase, f_from, f_to):
  """Map a stride phase with stance fraction f_from onto one with f_to."""
  return np.where(phase < f_from, phase * f_to / f_from,
                  f_to + (phase - f_from) * (1 - f_to) / (1 - f_from))


def lookup(c, ph):
  x = (ph % 1.0) * BINS
  i0 = np.floor(x).astype(int) % BINS
  fr = (x - np.floor(x))[:, None]
  return (1 - fr) * c[i0] + fr * c[(i0 + 1) % BINS]


def direction_ok(src, sgn, q, yaw, dt):
  def keep(a, b):
    T = (b - a) * dt
    if not 0.7 < T < 3.0:
      return False
    d = q[b, :2] - q[a, :2]
    c, s = np.cos(yaw[a]), np.sin(yaw[a])
    vx, vy = (d[0] * c + d[1] * s) / T, (-d[0] * s + d[1] * c) / T
    return ((src == "FW" and vx > 0.15) or (src == "BW" and vx < -0.1)
            or (src == "SW" and vy * sgn > 0.05 and abs(vx) < 0.15))
  return keep


def main():
  alpha, out = float(sys.argv[1]), sys.argv[2]
  raise_m = float(sys.argv[3]) if len(sys.argv) > 3 else 0.015
  m = rt.model()
  d = mujoco.MjData(m)
  jn = [m.joint(j).name for j in range(1, m.njnt)]
  side = {s: [jn.index(f"{s}_{n}") for n in LEG] for s in ("L", "R")}
  rest = [i for i in range(len(jn)) if not any(i in v for v in side.values())]
  ank = {s: m.body(f"{s}_ANKLE_P_LINK").id for s in ("L", "R")}
  z = dict(np.load("docs/bwc_motion_v3.npz"))
  fps = float(z["fps"][0])
  cs = list(z["clip_starts"]) + [len(z["joint_pos"])]
  q0 = np.concatenate([z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]], 1).astype(float)
  q = q0.copy()
  leg_dofs = [m.joint(jn[i]).dofadr[0] for v in side.values() for i in v]
  lo_l = np.array([m.jnt_range[m.dof_jntid[j], 0] for j in leg_dofs])
  hi_l = np.array([m.jnt_range[m.dof_jntid[j], 1] for j in leg_dofs])
  kL, kR = jn.index("L_KNEE_P"), jn.index("R_KNEE_P")
  for k, (a, b) in enumerate(zip(cs[:-1], cs[1:])):
    if k not in CLIP_DIR:
      continue
    src, sgn = CLIP_DIR[k]
    # human strides
    hz = np.load(f"{br.CACHE}/Neutral_{src}.npz")
    hq, hdt, hsole = hz["qpos"], float(hz["dt"]), hz["sole_z"]
    hpel, hyaw = pelvis_signals(hq)
    keep_h = direction_ok(src, sgn, hq, hyaw, hdt)
    # BWC strides of this clip
    bq = q0[a:b]
    bpel, _ = pelvis_signals(bq)
    zf = {s: z["body_pos_w"][a:b, ank[s] - 1, 2] for s in ("L", "R")}
    base = min(v.min() for v in zf.values())
    cyc = {}
    for si, s in enumerate(("L", "R")):
      htd, hlo = foot_events(hsole[:, si] > 0.01)
      btd, blo = foot_events(zf[s] > base + 0.01)
      ch, fh = stride_cycles(hq[:, 7:][:, side[s]], htd, hlo, keep_h)
      cb, fb = stride_cycles(bq[:, 7:][:, side[s]], btd, blo, lambda a_, b_: True)
      cyc[s] = (smooth_cycle(np.median(ch, 0)), smooth_cycle(np.median(cb, 0)), fh, fb, btd, len(ch))
    htd, hlo = foot_events(hsole[:, 0] > 0.01)
    btd, blo = foot_events(zf["L"] > base + 0.01)
    up_h, fh = stride_cycles(np.concatenate([hq[:, 7:][:, rest], hpel], 1), htd, hlo, keep_h)
    up_b, fb = stride_cycles(np.concatenate([bq[:, 7:][:, rest], bpel], 1), btd, blo, lambda a_, b_: True)
    up_h, up_b = smooth_cycle(np.median(up_h, 0)), smooth_cycle(np.median(up_b, 0))

    def phase_and_w(td):
      ph = np.full(b - a, np.nan)
      w = np.zeros(b - a)
      for s_, (t0, t1) in enumerate(zip(td[:-1], td[1:])):
        ph[t0:t1] = (np.arange(t0, t1) - t0) / (t1 - t0)
        w[t0:t1] = ph[t0:t1] if s_ == 0 else (1 - ph[t0:t1]) if s_ == len(td) - 2 else 1
      return ph, w

    phL, wL = phase_and_w(btd)
    walk = ~np.isnan(phL)
    # upper body + pelvis (left foot phase)
    dU = lookup(up_h - up_h.mean(0), warp(np.nan_to_num(phL), fb, fh)) - lookup(up_b - up_b.mean(0), np.nan_to_num(phL))
    dU *= (alpha * wL)[:, None]
    q[a:b, 7 + np.array(rest)] += dU[:, :len(rest)]
    q[a:b, 2] += dU[:, len(rest)] + raise_m * wL
    R = Rot.from_quat(q[a:b, 3:7], scalar_first=True)
    q[a:b, 3:7] = (R * Rot.from_rotvec(dU[:, len(rest) + 1:])).as_quat(scalar_first=True)
    # legs (own foot phase, warped)
    for s in ("L", "R"):
      ch, cb, fh_, fb_, td_s, n = cyc[s]
      ph_s, w_s = phase_and_w(td_s)
      ok = ~np.isnan(ph_s)
      dL = lookup(ch - ch.mean(0), warp(np.nan_to_num(ph_s), fb_, fh_)) - lookup(cb - cb.mean(0), np.nan_to_num(ph_s))
      q[a:b, 7 + np.array(side[s])] += (alpha * np.minimum(w_s, wL) * ok)[:, None] * dL
      amp = np.degrees(np.ptp(ch, 0))
      print(f"clip {k} ({src}) jambe {s}: {n} foulees humaines, appui {fh_:.2f} (BWC {fb_:.2f}), "
            f"genou humain crete a crete {amp[3]:.0f} deg")
    # IK per frame: stance foot on the BWC pose, swing foot human-shaped, landing on the
    # BWC foothold. Pass 1 finds how far toward the blended pelvis each frame can go
    # (feet reached, knee >= 8 deg); that fraction is smoothed in time (moving min over
    # +-0.25 s, then a 0.2 s mean) so the pelvis never jumps; pass 2 solves with it.
    frames = np.where(walk | (wL > 0))[0]
    tgt_all = q[a:b].copy()

    def targets_at(i, x):
      d.qpos[:] = q0[a + i]
      mujoco.mj_kinematics(m, d)
      bw = {s: (d.xpos[ank[s]].copy(), d.xmat[ank[s]].reshape(3, 3).copy()) for s in ("L", "R")}
      d.qpos[:] = x
      mujoco.mj_kinematics(m, d)
      tg = {}
      for s in ("L", "R"):
        up = zf[s] > base + 0.01
        if not up[i]:
          tg[ank[s]] = bw[s]
          continue
        i0 = i
        while i0 > 0 and up[i0 - 1]:
          i0 -= 1
        i1 = i
        while i1 < len(up) - 1 and up[i1 + 1]:
          i1 += 1
        u = (i - i0 + 0.5) / max(i1 - i0 + 1, 1)
        # 1 at lift-off, 0 at u = 0.375, back to 1 at u = 0.75: the last quarter of the
        # swing is the BWC's own, so the reference lands no faster than the BWC
        s_bw = 1.0 - np.sin(np.pi * min(u / 0.75, 1.0)) ** 2
        p_h, R_h = d.xpos[ank[s]].copy(), d.xmat[ank[s]].reshape(3, 3).copy()
        p_ = s_bw * bw[s][0] + (1 - s_bw) * p_h
        p_[2] = max(p_[2], bw[s][0][2], base)
        rv = Rot.from_matrix(bw[s][1] @ R_h.T).as_rotvec() * s_bw
        tg[ank[s]] = (p_, Rot.from_rotvec(rv).as_matrix() @ R_h)
      return tg

    def pelvis_at(i, f):
      x = tgt_all[i].copy()
      x[2] = q0[a + i, 2] + f * (tgt_all[i, 2] - q0[a + i, 2])
      R0 = Rot.from_quat(q0[a + i, 3:7], scalar_first=True)
      x[3:7] = (R0 * Rot.from_rotvec(f * (R0.inv() * Rot.from_quat(tgt_all[i, 3:7], scalar_first=True))
                                     .as_rotvec())).as_quat(scalar_first=True)
      return x

    freq = np.ones(b - a)
    for i in frames:
      for f in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0):
        x = pelvis_at(i, f)
        x, err = br.leg_ik(m, d, x, targets_at(i, x), leg_dofs, iters=30, limits=(lo_l, hi_l))
        if err < 0.003 and min(x[7 + kL], x[7 + kR]) > np.radians(8):
          break
      freq[i] = f
    h = int(0.25 * fps)
    fmin = np.array([freq[max(0, i - h):i + h + 1].min() for i in range(b - a)])
    k_ = int(0.2 * fps)
    fs = np.convolve(np.pad(fmin, k_ // 2, mode="edge"), np.ones(k_) / k_, "valid")[: b - a]
    fs = np.minimum(fs, fmin + 0.15)
    worst = 0.0
    for i in frames:
      x = pelvis_at(i, fs[i])
      x, err = br.leg_ik(m, d, x, targets_at(i, x), leg_dofs, iters=30, limits=(lo_l, hi_l))
      q[a + i] = x
      worst = max(worst, err)
    print(f"clip {k}: residu IK des chevilles max {worst * 1000:.1f} mm, bassin applique moyen {fs[frames].mean():.2f} (min {fs[frames].min():.2f})")
  lo_, hi_ = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  clipped = (q[:, 7:] < lo_) | (q[:, 7:] > hi_)
  if clipped.any():
    print("butees atteintes :", sorted({jn[j] for j in np.where(clipped.any(0))[0]}))
  q[:, 7:] = np.clip(q[:, 7:], lo_, hi_)
  pos = np.zeros_like(z["body_pos_w"], dtype=float)
  quat = np.zeros_like(z["body_quat_w"], dtype=float)
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(m, d)
    pos[i], quat[i] = d.xpos[1:], d.xquat[1:]
  dt = 1.0 / fps
  jv, lv, av = np.zeros((len(q), 30)), np.zeros_like(pos), np.zeros_like(pos)
  for a, b in zip(cs[:-1], cs[1:]):
    jv[a:b] = np.gradient(q[a:b, 7:], dt, axis=0)
    lv[a:b] = np.gradient(pos[a:b], dt, axis=0)
    Rm = Rot.from_quat(quat[a:b].reshape(-1, 4), scalar_first=True).as_matrix().reshape(b - a, -1, 3, 3)
    dR = np.einsum("tbij,tbkj->tbik", np.gradient(Rm, dt, axis=0), Rm)
    av[a:b] = np.stack([dR[..., 2, 1], dR[..., 0, 2], dR[..., 1, 0]], -1)
  # untouched clips keep their original velocities exactly
  for k, (a, b) in enumerate(zip(cs[:-1], cs[1:])):
    if k not in CLIP_DIR:
      jv[a:b], lv[a:b], av[a:b] = z["joint_vel"][a:b], z["body_lin_vel_w"][a:b], z["body_ang_vel_w"][a:b]
  o = dict(z)
  o.update(joint_pos=q[:, 7:], joint_vel=jv, body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lv, body_ang_vel_w=av)
  o = {k_: v.astype(z[k_].dtype) for k_, v in o.items()}
  np.savez_compressed(out, **o)
  print("ecrit", out)


if __name__ == "__main__":
  main()
