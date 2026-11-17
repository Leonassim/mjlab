"""overhang.py: where the loaded sole sits on the step, mc_mujoco stairs runs (Leo 2026-11-17: on the way down the
foot lands with the edge under its middle).

  python3 overhang.py export <log.bin> <out.npz>          system python (mc_log_ui): pelvis pose, qIn, sole Fz
  .venv/bin/python overhang.py stats <label> <npz>...     mujoco FK, one line per run + a summary line

Geometry of mc_mujoco_stairs.sh (centre x 1.64, 3 steps, tread 0.27, platform 1.6); the riser height is stored in
the npz name (_h0.05_). A foothold is a loaded stretch (Fz > 150 N) of one foot, read 0.1 s after touchdown.
  descent: pelvis past the platform centre, sole on a step or the platform (not the ground)
  climb:   pelvis before the centre, sole on a step or the platform
For each foothold: share of the sole underside (9 x 3 points of mc_mujoco's sole box) more than half a riser below
the step it stands on, and the 6-ray count of mjlab's stairs_soles (x -0.09/0/0.09, y +-0.045 from the ankle):
"straddle" = 3 or more rays out, the rtx4080 stairs_gait.py criterion.
"""
import sys

import numpy as np

CX, N, TREAD, PLAT = 1.64, 3, 0.27, 1.6
MODULE_ORDER = [
  "L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P", "CHEST_Y", "CHEST_P",
  "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P", "HEAD_Y", "HEAD_P",
  "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P", "L_ELBOW_Y", "L_WRIST_R", "L_WRIST_Y",
  "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", "R_ELBOW_Y", "R_WRIST_R", "R_WRIST_Y"]


def export(log, out):
  from mc_log_ui import read_log  # noqa: PLC0415
  d = read_log(log)
  g = lambda k: np.asarray(d[k], np.float32)  # noqa: E731
  armed = np.abs(np.asarray(d["NewRLQPController_RL_currentAction_0"])) > 0
  np.savez_compressed(out, t=g("t"), armed=armed,
                      p=np.stack([g(f"FloatingBase_position_{c}") for c in "xyz"], 1),
                      q=np.stack([g(f"FloatingBase_orientation_{c}") for c in "wxyz"], 1),
                      qin=np.stack([g(f"qIn_{i}") for i in range(len(MODULE_ORDER))], 1),
                      fz=np.stack([g("NewRLQPController_leftFootForceZ"), g("NewRLQPController_rightFootForceZ")], 1))


def height(x, y, h):
  dd = np.maximum(np.abs(x - CX), np.abs(y))
  k = N - np.ceil((dd - PLAT / 2) / TREAD)
  return np.where(dd <= PLAT / 2, N * h, np.where(k > 0, k * h, 0.0))


def stats(label, files):
  import mujoco  # noqa: PLC0415
  m = mujoco.MjModel.from_xml_path("/home/lmoussafir/install/share/mc_mujoco/RHPS1/xml/RHPS1main.xml")
  dat = mujoco.MjData(m)
  adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODULE_ORDER]
  feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
  sole = []
  for b in feet:  # the sole box: the colliding box of the ankle-pitch body
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
         and m.geom_contype[i]][0]
    sx, sy, sz = m.geom_size[g]
    px, py, pz = m.geom_pos[g]
    gx, gy = np.meshgrid(np.linspace(-sx, sx, 9), np.linspace(-sy, sy, 3), indexing="ij")
    sole.append(np.stack([px + gx.ravel(), py + gy.ravel(), np.full(gx.size, pz - sz)], 1))
  rays = np.array([(x, y) for x in (-0.09, 0.0, 0.09) for y in (-0.045, 0.045)])
  allr = {"descent": [], "climb": []}
  for f in files:
    h = float(f.split("_h")[-1].split("_")[0])
    z = np.load(f)
    t, armed, P, Q, qin, fz = z["t"], z["armed"], z["p"], z["q"], z["qin"], z["fz"]
    dt = float(np.median(np.diff(t)))
    fall = np.where(P[:, 2] < 0.55)[0]
    end = fall[0] if len(fall) else len(t)
    res = {"descent": [], "climb": []}
    for j in range(2):
      on = (fz[:, j] > 150) & armed
      on[end:] = False
      starts = np.flatnonzero(on[1:] & ~on[:-1]) + 1
      for s in starts:
        e = s
        while e < len(on) and on[e]:
          e += 1
        i = min(s + int(0.1 / dt), (s + e) // 2)
        dat.qpos[:3] = P[i]
        dat.qpos[3:7] = Q[i] * np.array([1, -1, -1, -1])  # mc_rtc logs the inverse rotation
        dat.qpos[adr] = qin[i]
        mujoco.mj_kinematics(m, dat)
        R = dat.xmat[feet[j]].reshape(3, 3)
        w = dat.xpos[feet[j]] + sole[j] @ R.T
        hs = height(w[:, 0], w[:, 1], h)
        top = hs.max()
        if top < 0.5 * h:  # on the ground
          continue
        out = (hs < top - 0.5 * h).mean()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s_ = np.cos(yaw), np.sin(yaw)
        rx = dat.xpos[feet[j]][0] + c * rays[:, 0] - s_ * rays[:, 1]
        ry = dat.xpos[feet[j]][1] + s_ * rays[:, 0] + c * rays[:, 1]
        nray = int((height(rx, ry, h) < top - 0.5 * h).sum())
        side = "descent" if P[i, 0] > CX else "climb"
        res[side].append((out, nray))
    for side in res:
      allr[side] += res[side]
    dsc = res["descent"]
    print(f"  {f.rsplit('/', 1)[-1][:-4]}: descent footholds {len(dsc)}, sole out {np.mean([r[0] for r in dsc]) if dsc else float('nan'):.0%}, "
          f"straddle {sum(r[1] >= 3 for r in dsc)}/{len(dsc)} | climb {len(res['climb'])}")
  parts = []
  for side in ("descent", "climb"):
    r = allr[side]
    if r:
      parts.append(f"{side} {len(r)} footholds: sole out mean {np.mean([x[0] for x in r]):.0%}, >= half "
                   f"{np.mean([x[0] >= 0.5 for x in r]):.0%}, straddle (>= 3/6 rays) {np.mean([x[1] >= 3 for x in r]):.0%}")
    else:
      parts.append(f"{side}: none")
  print(f"OVERHANG {label}: " + " | ".join(parts))


if __name__ == "__main__":
  if sys.argv[1] == "export":
    export(sys.argv[2], sys.argv[3])
  else:
    stats(sys.argv[2], sys.argv[3:])
