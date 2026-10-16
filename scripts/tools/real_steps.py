"""Real-robot steps from a log: swing ankle lift and stance-foot yaw slip, per command.

  python3 scripts/tools/real_steps.py extract <log.bin> <out.npz>     (system python, mc_log_ui)
  uv run python scripts/tools/real_steps.py analyze <out.npz> [...]   (mjlab venv, MuJoCo)
  analyze-target: same kinematics on the network's target instead of the encoders

Forward kinematics of the
MuJoCo RHPS1 model from the encoders (qIn, matched to the policy observation)
and the IMU orientation. Per single-support phase (other foot below 20 N, this
one above 100 N): the swing ankle's peak height above the stance ankle -- the
same quantity lift_probe.py measures in sim -- and the stance foot's world yaw
change, which is zero for a planted foot.
"""

import sys

import numpy as np

J = ("CHEST_Y,CHEST_P,HEAD_Y,HEAD_P,L_SHOULDER_P,L_SHOULDER_R,L_SHOULDER_Y,L_ELBOW_P,L_ELBOW_Y,"
     "L_WRIST_R,L_WRIST_Y,R_SHOULDER_P,R_SHOULDER_R,R_SHOULDER_Y,R_ELBOW_P,R_ELBOW_Y,R_WRIST_R,"
     "R_WRIST_Y,L_CROTCH_Y,L_CROTCH_R,L_CROTCH_P,L_KNEE_P,L_ANKLE_R,L_ANKLE_P,R_CROTCH_Y,R_CROTCH_R,"
     "R_CROTCH_P,R_KNEE_P,R_ANKLE_R,R_ANKLE_P").split(",")
Q0 = np.array([0, 0, 0, 0, 0.2617990077, 0.1745329946, -0.08726599813, -0.5235990286, 0, 0, 0,
               0.2617990077, -0.1745329946, 0.08726599813, -0.5235990286, 0, 0, 0, -0.01053300034,
               0.0287870001, -0.2711910009, 0.6220210195, -0.03065199964, -0.3506790102,
               0.01053300034, -0.0287870001, -0.2711910009, 0.6220210195, 0.03065199964,
               -0.3506790102])


def extract(log, out):
  import mc_log_ui

  d = mc_log_ui.read_log(log)
  t = np.asarray(d["t"])
  ob = np.stack([np.asarray(d[f"NewRLQPController_RL_currentObservation_{i}"]) for i in range(126)], -1)
  armed = np.abs(ob[:, 20]) > 1e-6
  nq = sum(1 for k in d if k.startswith("qIn_"))
  qin = np.stack([np.asarray(d[f"qIn_{i}"]) for i in range(nq)], -1)
  qpol = ob[:, 21:51] + Q0
  A = np.where(armed)[0]
  m = [int(np.argmin([np.mean(np.abs(qin[A, c] - qpol[A, p])) for c in range(nq)])) for p in range(30)]
  quat = np.stack([np.asarray(d[f"Accelerometer_orientation_{c}"]) for c in "wxyz"], -1)
  forces = np.stack([np.asarray(d["LeftFootForceSensor_fz"]), np.asarray(d["RightFootForceSensor_fz"])], -1)
  cmd = np.sign(np.round(ob[:, 123:126] * (np.abs(ob[:, 123:126]) > 0.5))).astype(int)
  order = ("L_CROTCH_Y L_CROTCH_R L_CROTCH_P L_KNEE_P L_ANKLE_R L_ANKLE_P CHEST_Y CHEST_P R_CROTCH_Y "
           "R_CROTCH_R R_CROTCH_P R_KNEE_P R_ANKLE_R R_ANKLE_P HEAD_Y HEAD_P L_SHOULDER_P L_SHOULDER_R "
           "L_SHOULDER_Y L_ELBOW_P L_ELBOW_Y L_WRIST_R L_WRIST_Y R_SHOULDER_P R_SHOULDER_R R_SHOULDER_Y "
           "R_ELBOW_P R_ELBOW_Y R_WRIST_R R_WRIST_Y").split()
  rlq = np.stack([np.asarray(d[f"NewRLQPController_RL_q_{i}"]) for i in range(30)], -1)
  target = rlq[:, [order.index(n) for n in J]]  # network target, policy order
  np.savez_compressed(out, t=t, armed=armed, q=qin[:, m], target=target, quat=quat, forces=forces, cmd=cmd,
                      grav=ob[:, 18:21])


def analyze(paths):
  sys.argv = sys.argv[:1] + ["r", "c"]
  sys.path.insert(0, "scripts/tools")
  import gait_dynamics as g
  import mujoco

  M, D = g.M, g.D
  names = [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(M.nbody)]
  foot = [next(i for i, n in enumerate(names) if n.endswith(s)) for s in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
  qadr = [M.jnt_qposadr[g._jid(n)] for n in J]
  for p in paths:
    r = np.load(p)
    armed, q, quat, F, cmd = r["armed"], r["q"], r["quat"], r["forces"], r["cmd"]
    if _SRC == "target":
      q = r["target"]
    T = len(armed)
    fz, fyaw = np.zeros((T, 2)), np.zeros((T, 2))
    for k in np.where(armed)[0]:
      D.qpos[:] = 0
      w, x, y, z = quat[k]
      D.qpos[g.FREE_Q + 3:g.FREE_Q + 7] = [w, -x, -y, -z]  # mc_rtc logs the inverse rotation
      D.qpos[qadr] = q[k]
      mujoco.mj_kinematics(M, D)
      for s in range(2):
        fz[k, s] = D.xpos[foot[s], 2]
        R = D.xmat[foot[s]].reshape(3, 3)
        fyaw[k, s] = np.arctan2(R[1, 0], R[0, 0])
    print(f"== {p.split('/')[-1]}")
    rows = {}
    for s in range(2):
      me, other = F[:, s], F[:, 1 - s]
      ss = armed & (me > 100) & (other < 20)
      e = np.diff(np.r_[0, ss.astype(int), 0])
      for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]):
        if (b - a) * 0.005 < 0.15:
          continue
        c = tuple(int(v) for v in cmd[(a + b) // 2])
        rel = fz[a:b, 1 - s] - fz[a:b, s]
        lift = rel.max() - rel[0]
        yw = np.unwrap(fyaw[a:b, s])
        rows.setdefault(c, []).append((lift, abs(np.degrees(yw[-1] - yw[0]))))
    for c, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
      v = np.array(v)
      print(f"  cmd {c!s:12s} {len(v):3d} appuis simples: levee cheville mediane {100*np.median(v[:, 0]):4.1f} cm"
            f" (p10 {100*np.percentile(v[:, 0], 10):4.1f}), rotation pied d'appui mediane {np.median(v[:, 1]):4.1f} deg"
            f" (p90 {np.percentile(v[:, 1], 90):4.1f})")


def compliance(paths):
  """Pelvis tilt seen by the IMU vs predicted by the encoders with a flat, rigid stance foot.

  Gravity in the pelvis frame: measured = the policy's projected_gravity (IMU via
  the estimator); predicted = R_pelvis_foot * (0, 0, -1) from forward kinematics.
  Equal on a rigid robot with a flat foot; the excess is compliance the encoders
  do not see (gears, cylinders, soles) or the foot tilting on the ground.
  """
  sys.argv = sys.argv[:1] + ["r", "c"]
  sys.path.insert(0, "scripts/tools")
  import gait_dynamics as g
  import mujoco

  M, D = g.M, g.D
  names = [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(M.nbody)]
  foot = [next(i for i, n in enumerate(names) if n.endswith(s)) for s in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
  qadr = [M.jnt_qposadr[g._jid(n)] for n in J]
  for p in paths:
    r = np.load(p)
    armed, q, F, cmd, gm = r["armed"], r["q"], r["forces"], r["cmd"], r["grav"]
    print(f"== {p.split('/')[-1]}")
    rows = {}
    for s in range(2):
      ss = armed & (F[:, s] > 100) & (F[:, 1 - s] < 20)
      e = np.diff(np.r_[0, ss.astype(int), 0])
      for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]):
        if (b - a) * 0.005 < 0.2:
          continue
        gp = np.zeros((b - a, 3))
        for k in range(a, b):
          D.qpos[:] = 0
          D.qpos[g.FREE_Q + 3] = 1.0
          D.qpos[qadr] = q[k]
          mujoco.mj_kinematics(M, D)
          gp[k - a] = D.xmat[foot[s]].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
        meas = np.degrees(np.stack([np.arcsin(np.clip(-gm[a:b, 1], -1, 1)), np.arcsin(np.clip(gm[a:b, 0], -1, 1))], -1))
        pred = np.degrees(np.stack([np.arcsin(np.clip(-gp[:, 1], -1, 1)), np.arcsin(np.clip(gp[:, 0], -1, 1))], -1))
        c = tuple(int(v) for v in cmd[(a + b) // 2])
        dm, dp = meas - meas.mean(0), pred - pred.mean(0)
        rows.setdefault(c, []).append(np.r_[dm.std(0), dp.std(0), (dm - dp).std(0)])
    for c, v in sorted(rows.items(), key=lambda kv: -len(kv[1])):
      v = np.median(np.array(v), 0)
      print(f"  cmd {c!s:12s} {len(rows[c]):3d} appuis: roulis IMU {v[0]:.2f} / encodeurs {v[2]:.2f} / ecart {v[4]:.2f} deg"
            f" | tangage IMU {v[1]:.2f} / encodeurs {v[3]:.2f} / ecart {v[5]:.2f} deg (ecart-type pendant l'appui)")


if __name__ == "__main__":
  if sys.argv[1] == "compliance":
    compliance(sys.argv[2:])
    sys.exit()
  if sys.argv[1] == "extract":
    extract(sys.argv[2], sys.argv[3])
  else:
    _SRC = "target" if sys.argv[1] == "analyze-target" else "q"
    analyze(sys.argv[2:])
