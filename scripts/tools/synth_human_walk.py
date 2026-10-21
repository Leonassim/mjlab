"""Straight human walking for RHPS1, synthesized from the mean 100STYLE Neutral stride.

  uv run python scripts/tools/synth_human_walk.py <out.npz> [<leg_amp>] [<slow>]

The 100STYLE actor never walks straight for 4 s, and straightening the paths would
slide the feet. So the mean stride (all joints, pelvis roll/pitch/yaw oscillation,
flat-foot contact schedule, ankle lift; FW forward, BW backward; median over ~80
strides, 6 harmonics) is replayed at the human cadence slowed `slow` times (default 2).
A foot is in stance only while flat (heel strike and toe-off are swing for the robot's
flat feet). The pelvis moves with the stance feet of the cycle, 1 cm below the lowest
stance requirement (no straight knee); stance feet stay flat where they landed; swing
feet lift at the human ankle height, travel minimum-jerk between footholds and take
the human foot orientation mid-swing; the soles never go below the ground. Legs are
solved by IK on both ankles.

Gait initiation and termination: the stride amplitude ramps from the BWC's standing
pose (feet together, the deployment q0 within a few degrees) to the full human stride
over 1.5 strides, and back. Arms keep the BWC's mean posture (clear of the QP
collision zones) with the human swing on top; legs take the human mean posture and
`leg_amp` (default 1) of its oscillation.

Clips: [human forward] [BWC left] [human backward] [BWC right] [BWC yaw left]
[BWC yaw right], so every command keeps a single style. bwc_motion_v3.npz format.
"""

import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import blend_global as bg  # noqa: E402
import blend_reference as br  # noqa: E402
import retarget_lafan as rt  # noqa: E402

FPS = 200.0
BINS = bg.BINS
M = rt.model()
JN = [M.joint(j).name for j in range(1, M.njnt)]
LEGS = [i for i, n in enumerate(JN) if any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]
ARMS = [i for i, n in enumerate(JN) if any(t in n for t in ("SHOULDER", "ELBOW", "WRIST"))]
ANK = {s: M.body(f"{s}_ANKLE_P_LINK").id for s in ("L", "R")}
LEG_DOFS = [M.joint(JN[j]).dofadr[0] for j in LEGS]
LEG_LIM = (np.array([M.jnt_range[M.dof_jntid[j], 0] for j in LEG_DOFS]),
           np.array([M.jnt_range[M.dof_jntid[j], 1] for j in LEG_DOFS]))


def human_stride(src):
  z = np.load(f"{br.CACHE}/Neutral_{src}.npz")
  q, dt, sole = z["qpos"], float(z["dt"]), z["sole_z"]
  pel, yaw = bg.pelvis_signals(q)
  keep = bg.direction_ok(src, 1, q, yaw, dt)
  td, lo = bg.foot_events(sole[:, 0] > 0.01)
  # flat-foot stance: ankle within 1.2 cm of its lowest; heel strike and toe-off are
  # swing for a flat-footed robot
  d = mujoco.MjData(M)
  az = np.zeros((len(q), 2))
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(M, d)
    az[i] = d.xpos[[ANK["L"], ANK["R"]], 2]
  flat_down = (az < az.min(0) + 0.012) & (sole < 0.01)
  sig = np.concatenate([q[:, 7:], pel[:, 1:], flat_down.astype(float), az - az.min(0)], 1)
  cyc, _ = bg.stride_cycles(sig, td, lo, keep)
  periods = [(b - a) * dt for a, b in zip(td[:-1], td[1:], strict=True) if keep(a, b)]
  c = np.median(cyc, 0)
  joints = bg.smooth_cycle(c[:, :30])
  rot = bg.smooth_cycle(c[:, 30:33] - c[:, 30:33].mean(0))
  stance = c[:, 33:35] > 0.5
  lift = np.maximum(bg.smooth_cycle(c[:, 35:37]) - 0.012, 0.0)  # ankle height above flat-down
  return joints, rot, stance, lift, float(np.median(periods)), len(cyc)


def bwc_stand():
  z = np.load("docs/bwc_motion_v3.npz")
  i = int(5 * FPS)  # well inside the standing prologue of clip 0
  q = np.concatenate([z["body_pos_w"][i, 0], z["body_quat_w"][i, 0], z["joint_pos"][i]])
  return q.astype(float)


def smooth(x):
  x = float(np.clip(x, 0, 1))
  return x * x * (3 - 2 * x)


def foot_pose(q):
  d = mujoco.MjData(M)
  d.qpos[:] = q
  mujoco.mj_kinematics(M, d)
  return {s: (d.xpos[ANK[s]].copy(), d.xmat[ANK[s]].reshape(3, 3).copy()) for s in ("L", "R")}


def synth(src, walk_s, leg_amp, slow, stand_s=4.0, ramp_strides=1.5):
  joints, rot, stance, lift, period, n = human_stride(src)
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
  a = np.clip((t - stand_s) / ramp, 0, 1) * np.clip((total - stand_s - t) / ramp, 0, 1)
  a = a * a * (3 - 2 * a)  # smoothstep
  ph = np.where(t > stand_s, (t - stand_s) / T, 0.0) % 1.0
  cj = bg.lookup(cyc, ph)
  cr = bg.lookup(rot, ph)
  st = bg.lookup(stance.astype(float), ph) > 0.5
  cl = bg.lookup(lift, ph) * a[:, None]
  st[a < 0.15] = True  # small steps near the stand: both feet down
  qj = q_stand[None] + a[:, None] * (cj - q_stand[None])
  heading = Rot.from_quat(qs[3:7], scalar_first=True).as_euler("ZYX")[0]
  R0 = Rot.from_euler("Z", heading)
  # pelvis orientation: heading x human oscillation (scaled by the amplitude)
  Rp = [R0 * Rot.from_rotvec(a[i] * cr[i]) * (R0.inv() * Rot.from_quat(qs[3:7], scalar_first=True)) for i in range(len(t))]
  # Pelvis and footholds, both continuous. r_s = ankle position relative to the pelvis
  # from the cycle's own kinematics; while a foot is down the pelvis moves by -dr_s
  # (mean of the two in double support), so it never jumps; a foot lands where the
  # cycle brings it and stays there, flat.
  d = mujoco.MjData(M)
  n_t = len(t)
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
  ground = np.array([foot_pose(qs)[s][0][2] for s in ("L", "R")])
  pel = np.zeros((n_t, 3))
  pel[0] = qs[:3]
  for i in range(1, n_t):
    on = st[i] & st[i - 1]
    step = -(rel[i] - rel[i - 1])[on].mean(0) if on.any() else np.zeros(3)
    pel[i] = pel[i - 1] + step
  # pelvis height: stance ankles on the ground
  for i in range(n_t):  # lowest stance requirement, 1 cm lower: no straight knee
    on = st[i]
    pel[i, 2] = np.min(ground[on] - rel[i, on, 2]) - 0.01 * a[i] if on.any() else pel[i - 1, 2]
  k = int(0.1 * FPS)
  for c in range(3):  # 0.1 s smoothing, the double-support switches step the velocity
    pel[:, c] = np.convolve(np.pad(pel[:, c], k // 2, mode="edge"), np.ones(k) / k, "valid")[:n_t]
  # footholds: where the foot is at each touchdown, held flat during stance
  flat = {s: foot_pose(qs)[s][1] for s in ("L", "R")}
  d.qpos[:] = qs
  mujoco.mj_kinematics(M, d)
  sole0 = rt.sole_min_z(M, d)
  hold = np.zeros((n_t, 2, 3))
  for si in range(2):
    cur = qs[:3] + rel[0, si]
    cur[2] = ground[si]
    for i in range(n_t):
      if st[i, si] and (i == 0 or not st[i - 1, si]):
        cur = pel[i] + rel[i, si]
        cur[2] = ground[si]
      hold[i, si] = cur
  q = np.zeros((n_t, 37))
  worst = 0.0
  for i in range(n_t):
    x = qk[i].copy()
    x[:3] = pel[i]
    if i > 0:  # warm start
      x[np.array(LEG_DOFS) + 1] = q[i - 1, np.array(LEG_DOFS) + 1]
    d.qpos[:] = x
    mujoco.mj_kinematics(M, d)
    tg = {}
    for si, s in enumerate(("L", "R")):
      if st[i, si]:
        tg[ANK[s]] = (hold[i, si], flat[s])
      else:
        # swing: lifts flat at the human ankle height, minimum-jerk from foothold to
        # foothold, human foot orientation mid-swing
        i0 = i
        while i0 > 0 and not st[i0 - 1, si]:
          i0 -= 1
        i1 = i
        while i1 < n_t - 1 and not st[i1 + 1, si]:
          i1 += 1
        u = (i - i0 + 0.5) / max(i1 - i0 + 1, 1)
        p0, p1 = hold[max(i0 - 1, 0), si], hold[min(i1 + 1, n_t - 1), si]
        sx = np.clip((u - 0.05) / 0.9, 0, 1)
        p_t = p0 + sx ** 3 * (10 - 15 * sx + 6 * sx * sx) * (p1 - p0)
        p_t[2] = ground[si] + cl[i, si] * smooth(min(u, 1 - u) / 0.15)
        w = 1 - smooth(min(u, 1 - u) / 0.35)
        rv = Rot.from_matrix(flat[s] @ Rc[i, si].T).as_rotvec() * w
        tg[ANK[s]] = (p_t, Rot.from_rotvec(rv).as_matrix() @ Rc[i, si])
    x, err = br.leg_ik(M, d, x, tg, LEG_DOFS, iters=25, limits=LEG_LIM)
    for _ in range(3):  # a pitched swing foot near the ground: raise it until the sole clears
      sz = rt.sole_min_z(M, d)
      low = {si: sole0[si] - sz[si] for si in range(2) if not st[i, si] and sz[si] < sole0[si] - 1e-4}
      if not low:
        break
      for si, dz in low.items():
        b = ANK["LR"[si]]
        tg[b] = (tg[b][0] + np.array([0.0, 0.0, dz]), tg[b][1])
      x, err = br.leg_ik(M, d, x, tg, LEG_DOFS, iters=25, limits=LEG_LIM)
    worst = max(worst, err)
    q[i] = x
  print(f"  {src}: residu IK des pieds max {worst * 1000:.1f} mm")
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
