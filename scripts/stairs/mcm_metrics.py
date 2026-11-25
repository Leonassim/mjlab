"""mcm_metrics.py <mc_rtc .bin>... : flat mc_mujoco bench (headless.sh + RLQP_BENCH_VEL_DUR_S), slot 18 as the yardstick.

Walk window: from 2 s after the command comes on to its release. Stop window: 3 s (slip), 5 s (steps, still).
  speed   body-frame mean vx vy wz over the walk window (pelvis ground truth, FloatingBase)
  roll    pelvis roll p95 (deg); hike = mean roll toward the swing side in single support (deg)
  step    landing to landing median (s), single support share, foot lift (max ankle rise per swing, median, cm),
          swing knee minus stance knee (deg)
  stop    net travel of each loaded sole per stance (cm), landings after the release, time until still (s)
Slot 18 (2026-11-19, vx 0.15): roll 2.2, step 1.03 s, single support 96 %, lift ~5.6 cm, knee +21, net slip 1.6 cm.
  .venv/bin/python scripts/stairs/mcm_metrics.py ~/logs/mc_rtc/mc-control-NewRLQPController-*.bin
"""

import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

# mc_log_ui lives in the system python (with PyQt5), mujoco in the mjlab venv: the system python reads the log.
_READ = """
import sys, numpy as np, mc_log_ui
l = mc_log_ui.read_log(sys.argv[1])
keep = ("t", "FloatingBase_position_", "FloatingBase_orientation_", "qIn_", "NewRLQPController_velCmd_",
        "NewRLQPController_RL_currentAction_", "LeftFootForceSensor_fz", "RightFootForceSensor_fz",
        "NewRLQPController_leftFoot", "NewRLQPController_rightFoot")
np.savez(sys.argv[2], **{k: np.asarray(v, float) for k, v in l.items() if k == "t" or k.startswith(keep[1:])})
"""

ROBOT = ["L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P", "CHEST_Y", "CHEST_P",
         "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P", "HEAD_Y", "HEAD_P"] + \
        [f"{s}_{j}" for s in "LR" for j in ("SHOULDER_P", "SHOULDER_R", "SHOULDER_Y", "ELBOW_P", "ELBOW_Y", "WRIST_R", "WRIST_Y")]
XML = "/home/lmoussafir/mjlab-rhps1/src/mjlab/asset_zoo/robots/RHPS1/xmls/RHPS1main.xml"
m = mujoco.MjModel.from_xml_path(XML)
dd = mujoco.MjData(m)
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ROBOT]
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
KNEE = [ROBOT.index("L_KNEE_P"), ROBOT.index("R_KNEE_P")]


def load(f):
  with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "log.npz")
    subprocess.run(["/usr/bin/python3", "-c", _READ, f, out], check=True, stderr=subprocess.DEVNULL)
    l = dict(np.load(out))
  g = lambda k: l[k]
  pos = np.stack([g(f"FloatingBase_position_{a}") for a in "xyz"], 1)
  quat = np.stack([g(f"FloatingBase_orientation_{a}") for a in "wxyz"], 1)
  q = np.stack([g(f"qIn_{i}") for i in range(30)], 1)
  vc = np.stack([g(f"NewRLQPController_velCmd_{a}") for a in "xyz"], 1)
  a = np.stack([g(f"NewRLQPController_RL_currentAction_{i}") for i in range(30)], 1)
  imp = []
  for side in ("left", "right"):
    c = g(f"NewRLQPController_{side}FootContact")
    vz = np.abs(g(f"NewRLQPController_{side}FootImpactVel_z"))
    imp.append((c, vz))
  return g("t"), pos, quat, q, vc, a, g("LeftFootForceSensor_fz"), g("RightFootForceSensor_fz"), imp


print("run                 | speed vx    vy    wz  | roll p95 hike | step s  ss %  lift cm knee | td vz p90  Fz p90 | stop net cm steps still s")
for f in sys.argv[1:]:
  t, pos, quat, q, vc, act, fzL, fzR, imp = load(f)
  n = len(t)
  ft, roll, yaw = np.zeros((n, 2, 3)), np.zeros(n), np.zeros(n)
  for i in range(n):
    dd.qpos[:3] = pos[i]
    w, x, y, z = quat[i]
    dd.qpos[3:7] = [w, -x, -y, -z]  # mc_rtc logs the inverse rotation
    dd.qpos[adr] = q[i]
    mujoco.mj_kinematics(m, dd)
    ft[i] = dd.xpos[feet]
    R = dd.xmat[1].reshape(3, 3)
    roll[i] = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw[i] = np.arctan2(R[1, 0], R[0, 0])
  armed = np.abs(act).sum(1) > 0
  ta = t[np.argmax(armed)]
  fell = bool(((pos[:, 2] < 0.55) & (t > ta)).any())
  on = np.where(np.abs(vc).sum(1) > 1e-3)[0]
  if len(on) == 0:
    print(f"{f}: no command logged"); continue
  t_on, t_off = t[on[0]], t[on[-1]]
  wk = (t >= t_on + 2) & (t < t_off)
  dt = np.gradient(t)
  v = np.gradient(pos[:, :2], t, axis=0)
  vb = np.stack([v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw), -v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw),
                 np.gradient(np.unwrap(yaw), t)], 1)
  vm = vb[wk].mean(0)
  F = np.stack([fzL, fzR], 1)
  bw = np.median(F[(t > ta - 1) & (t < ta)].sum(1))
  ld = F > 30.0
  ss = (ld.sum(1) == 1) & wk
  sw = [(~ld[:, k]) & ld[:, 1 - k] & wk for k in range(2)]  # leg k swinging
  hike = np.mean(np.concatenate([-roll[sw[0]], roll[sw[1]]])) if ss.any() else np.nan  # >0: pelvis toward swing
  lands, lifts, knees = [], [], []
  for k in range(2):
    up = ~ld[:, k]
    idx = np.where(up & wk)[0]
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
      if len(seg) < 10:
        continue
      lands.append(t[seg[-1]])
      lifts.append(ft[seg, k, 2].max() - ft[seg[0], k, 2])
      st_k = ld[:, k] & wk
      knees.append(np.degrees(q[seg, KNEE[k]].max() - np.median(q[st_k, KNEE[k]])) if st_k.any() else np.nan)
  lands = np.sort(lands)
  gaps = np.diff(lands)
  gaps = gaps[gaps > 0.1]
  vzs, fpk = [], []
  for k, (c, vz) in enumerate(imp):
    rise = np.where((c[1:] > 0.5) & (c[:-1] < 0.5) & wk[1:])[0] + 1
    vzs += list(vz[np.minimum(rise + 1, n - 1)])
    fpk += [F[r:r + 40, k].max() for r in rise]
  st = (t >= t_off) & (t < t_off + 3)
  net = 0.0
  for k in range(2):
    idx = np.where(st & (F[:, k] > 0.25 * bw))[0]
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []:
      net += np.linalg.norm(ft[seg[-1], k, :2] - ft[seg[0], k, :2])
  steps, air = 0, [0.0, 0.0]
  for i in np.where((t >= t_off) & (t < t_off + 5))[0]:
    for k in range(2):
      if F[i, k] > 10:
        steps += air[k] >= 0.08
        air[k] = 0.0
      else:
        air[k] += dt[i]
  late = (t >= t_off) & (t < t_off + 5)
  fv = np.linalg.norm(np.gradient(ft[:, :, :2], t, axis=0), axis=2).max(1)
  mov = late & ((fv > 0.02) | (np.linalg.norm(v, axis=1) > 0.02))
  still = (t[mov].max() - t_off) if mov.any() else 0.0
  name = f.split("/")[-1].replace("mc-control-NewRLQPController-", "")[:-4]
  print(f"{name:19s} | {vm[0]:+.3f} {vm[1]:+.3f} {vm[2]:+.2f} | {np.percentile(np.abs(roll[wk]), 95):5.2f} {hike:+5.2f} | "
        f"{np.median(gaps) if len(gaps) else np.nan:5.2f} {100 * ss.sum() / max(wk.sum(), 1):4.0f} "
        f"{100 * np.median(lifts) if lifts else np.nan:6.1f} {np.nanmedian(knees) if knees else np.nan:+5.0f} | "
        f"{np.percentile(vzs, 90) if vzs else np.nan:9.3f} {np.percentile(fpk, 90) if fpk else np.nan:6.0f} | "
        f"{100 * net:7.1f} {steps:5d} {still:6.2f}{'  FELL' if fell else ''}")
