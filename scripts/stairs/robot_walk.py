"""robot_walk.py <mc_rtc .bin>: real-robot flat walking test, one line per command segment (no ground truth).

Armed = policy action non-zero. Segments = runs of the same command sign (vx, vy, wz) while armed. Per segment:
estimated pelvis displacement and yaw (MCWaiko world pose), body-frame speed, pelvis roll p95 from the IMU
orientation, landing to landing, single support share and foot lift (soles' Fz > 30 N, ankle height from kinematics on
the estimated pelvis), swing knee over stance, leg |qd| max and min sole load (the robot logs no joint torque).
  .venv/bin/python scripts/stairs/robot_walk.py ~/logs/rhps1_real/<date>/mc-control-*.bin
"""

import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

_READ = """
import sys, numpy as np, mc_log_ui
l = mc_log_ui.read_log(sys.argv[1])
keep = ("Observers_Pipeline_MCWaiko_FloatingBase_world_pose_", "qIn_", "NewRLQPController_velCmd_", "NewRLQPController_RL_currentAction_", "LeftFootForceSensor_fz",
        "RightFootForceSensor_fz", "NewRLQPController_joint_torque_Nm_")
np.savez(sys.argv[2], **{k: np.asarray(v, float) for k, v in l.items() if k == "t" or k.startswith(keep)})
"""
ROBOT = ["L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P", "CHEST_Y", "CHEST_P",
         "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P", "HEAD_Y", "HEAD_P"] + \
        [f"{s}_{j}" for s in "LR" for j in ("SHOULDER_P", "SHOULDER_R", "SHOULDER_Y", "ELBOW_P", "ELBOW_Y", "WRIST_R", "WRIST_Y")]
EFF = {"CROTCH_Y": 35, "CROTCH_R": 100, "CROTCH_P": 140, "KNEE_P": 108, "ANKLE_P": 65, "ANKLE_R": 45}
LEGS = [i for i, n in enumerate(ROBOT) if any(n.endswith(k) for k in EFF)]
m = mujoco.MjModel.from_xml_path("/home/lmoussafir/mjlab-rhps1/src/mjlab/asset_zoo/robots/RHPS1/xmls/RHPS1main.xml")
dd = mujoco.MjData(m)
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ROBOT]
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
KNEE = [ROBOT.index("L_KNEE_P"), ROBOT.index("R_KNEE_P")]

with tempfile.TemporaryDirectory() as d:
  out = os.path.join(d, "log.npz")
  subprocess.run(["/usr/bin/python3", "-c", _READ, sys.argv[1], out], check=True, stderr=subprocess.DEVNULL)
  z = dict(np.load(out))
t = z["t"]
n = len(t)
W = "Observers_Pipeline_MCWaiko_FloatingBase_world_pose_"  # ff_* is frozen on the robot (default Waiko pose)
pos = np.stack([z[f"{W}t{a}"] for a in "xyz"], 1)
quat = np.stack([z[f"{W}q{a}"] for a in "wxyz"], 1)
q = np.stack([z[f"qIn_{i}"] for i in range(30)], 1)
vc = np.stack([z[f"NewRLQPController_velCmd_{a}"] for a in "xyz"], 1)
act = np.stack([z[f"NewRLQPController_RL_currentAction_{i}"] for i in range(30)], 1)
F = np.stack([z["LeftFootForceSensor_fz"], z["RightFootForceSensor_fz"]], 1)
ft, roll, yaw = np.zeros((n, 2, 3)), np.zeros(n), np.zeros(n)
for i in range(n):
  dd.qpos[:3] = pos[i]
  w, x, y, zz = quat[i]
  dd.qpos[3:7] = [w, -x, -y, -zz]  # mc_rtc logs the inverse rotation
  dd.qpos[adr] = q[i]
  mujoco.mj_kinematics(m, dd)
  ft[i] = dd.xpos[feet]
  R = dd.xmat[1].reshape(3, 3)
  roll[i] = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
  yaw[i] = np.arctan2(R[1, 0], R[0, 0])
armed = np.abs(act).sum(1) > 0
qd = np.gradient(q, t, axis=0)
sgn = np.sign(np.round(vc, 3)) * armed[:, None]
key = sgn[:, 0] * 9 + sgn[:, 1] * 3 + sgn[:, 2]
edges = np.where(np.diff(key) != 0)[0] + 1
segs = np.split(np.arange(n), edges)
NAME = lambda s: "stop" if not s.any() else "+".join(f"{'+-'[int(v < 0)]}{a}" for v, a in zip(s, ("vx", "vy", "wz")) if v)
ta = np.where(armed)[0]
print(f"log {t[-1]:.0f} s | armed {armed.mean() * 100:.0f} % of the log, {len(np.split(ta, np.where(np.diff(ta) > 1)[0] + 1)) if len(ta) else 0} armed stretch(es)")
print(" t0 s  dur s  cmd          | disp m  dyaw deg  v body (x y wz) | roll p95 | step s ss % lift cm knee | qd max  Fz min")
for s in segs:
  if not armed[s[0]] or t[s[-1]] - t[s[0]] < 1.0:
    continue
  i0, i1 = s[0], s[-1]
  dur = t[i1] - t[i0]
  dp = pos[i1, :2] - pos[i0, :2]
  dyaw = np.degrees(np.unwrap(yaw[s])[-1] - np.unwrap(yaw[s])[0])
  y0 = yaw[i0]
  vb = (dp[0] * np.cos(y0) + dp[1] * np.sin(y0)) / dur, (-dp[0] * np.sin(y0) + dp[1] * np.cos(y0)) / dur, np.radians(dyaw) / dur
  ld = F[s] > 30
  ss = (ld.sum(1) == 1).mean()
  lands, lifts, knees = [], [], []
  for k in range(2):
    idx = np.where(~ld[:, k])[0]
    for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []:
      if len(g) < 8:
        continue
      lands.append(t[s][g[-1]])
      lifts.append(ft[s][g, k, 2].max() - ft[s][g[0], k, 2])
      st_k = ld[:, k]
      if st_k.any():
        knees.append(np.degrees(q[s][g, KNEE[k]].max() - np.median(q[s][st_k, KNEE[k]])))
  gaps = np.diff(np.sort(lands))
  gaps = gaps[gaps > 0.1]
  print(f"{t[i0]:6.1f} {dur:6.1f}  {NAME(sgn[i0]):12s} | {np.linalg.norm(dp):6.2f} {dyaw:+8.1f}  "
        f"{vb[0]:+.3f} {vb[1]:+.3f} {vb[2]:+.2f} | {np.percentile(np.abs(roll[s] - np.median(roll[s])), 95):8.2f} | "
        f"{np.median(gaps) if len(gaps) else np.nan:6.2f} {100 * ss:4.0f} {100 * np.median(lifts) if lifts else np.nan:7.1f} "
        f"{np.nanmedian(knees) if knees else np.nan:+4.0f} | {np.abs(qd[s][:, LEGS]).max():6.2f} {F[s].min():7.0f}")
tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(roll)), -1, 1)))
print(f"max |roll| while armed {np.abs(roll[armed]).max():.1f} deg; pelvis z range armed {pos[armed, 2].min():.2f}-{pos[armed, 2].max():.2f} m")
