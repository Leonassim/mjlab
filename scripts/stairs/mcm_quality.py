"""mcm_quality.py <mc_rtc .bin> [label]: gait quality Léo flagged on the robot (2026-11-27), mc_mujoco or real log.

Windows: walk = 2 s after the command comes on to its release; stop = release to +3 s; rest = +3 to +6 s; flat =
the 0.5 s before arming (feet flat on the floor, the reference for sole tilt and toe-out).
  jerk    leg target change per policy step (RL_q, deg: mean, p99) and leg joint acceleration rms (rad/s^2, 20 ms
          smoothing), walk window
  land    sole tilt at touchdown vs flat (pitch + = toe down, roll), median |.| and p90; toe-out change vs flat (deg)
  slip    horizontal travel of each sole while carrying > 30 % BW in the first 6 s of each stop (world frame), and the
          path of the foot-to-foot vector while both carry > 15 % BW (blind to estimator drift: use it on the robot)
  rest    sole tilt vs flat per foot (deg), load share L/R and the lighter foot's load (N)
Base pose: FloatingBase (mc_mujoco ground truth) when it moves, MCWaiko otherwise (real robot).
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
keep = ("FloatingBase_position_", "FloatingBase_orientation_", "Observers_Pipeline_MCWaiko_FloatingBase_world_pose_",
        "qIn_", "NewRLQPController_velCmd_", "NewRLQPController_RL_currentAction_", "NewRLQPController_RL_q_",
        "LeftFootForceSensor_fz", "RightFootForceSensor_fz")
np.savez(sys.argv[2], **{k: np.asarray(v, float) for k, v in l.items() if k == "t" or k.startswith(keep)})
"""
ROBOT = ["L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P", "CHEST_Y", "CHEST_P",
         "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P", "HEAD_Y", "HEAD_P"]
LEGS = [i for i, n in enumerate(ROBOT) if n[2:] in ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE_P", "ANKLE_R", "ANKLE_P")]
m = mujoco.MjModel.from_xml_path("/home/lmoussafir/mjlab-rhps1/src/mjlab/asset_zoo/robots/RHPS1/xmls/RHPS1main.xml")
dd = mujoco.MjData(m)
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ROBOT]
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
BW = float(sum(m.body_mass)) * 9.81

with tempfile.TemporaryDirectory() as d:
  out = os.path.join(d, "l.npz")
  subprocess.run(["/usr/bin/python3", "-c", _READ, sys.argv[1], out], check=True, stderr=subprocess.DEVNULL)
  z = dict(np.load(out))
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(sys.argv[1])
t = z["t"]
n = len(t)
dt = float(np.median(np.diff(t)))
if "FloatingBase_position_x" in z and np.ptp(z["FloatingBase_position_x"]) > 0.05:
  pos = np.stack([z[f"FloatingBase_position_{a}"] for a in "xyz"], 1)
  quat = np.stack([z[f"FloatingBase_orientation_{a}"] for a in "wxyz"], 1)
else:
  W = "Observers_Pipeline_MCWaiko_FloatingBase_world_pose_"
  pos = np.stack([z[f"{W}t{a}"] for a in "xyz"], 1)
  quat = np.stack([z[f"{W}q{a}"] for a in "wxyz"], 1)
q = np.stack([z[f"qIn_{i}"] for i in range(16)], 1)
qt = np.stack([z[f"NewRLQPController_RL_q_{i}"] for i in range(16)], 1)
act = np.stack([z[f"NewRLQPController_RL_currentAction_{i}"] for i in range(30)], 1)
armed = np.abs(act).sum(1) > 0
F = np.stack([z["LeftFootForceSensor_fz"], z["RightFootForceSensor_fz"]], 1)
vc = np.stack([z[f"NewRLQPController_velCmd_{a}"] for a in "xyz"], 1)
on = np.where(np.abs(vc).sum(1) > 1e-3)[0]
if not armed.any() or not len(on):
  sys.exit(f"{label}: never armed or no command")
ta, t0, t1 = t[armed][0], t[on[0]], t[on[-1]]

# Kinematics every 5 ms.
step = max(1, int(round(0.005 / dt)))
idx = np.arange(0, n, step)
fp = np.zeros((len(idx), 2, 3))
tilt = np.zeros((len(idx), 2, 2))  # pitch (+ toe down), roll
fyaw = np.zeros((len(idx), 2))
for k, i in enumerate(idx):
  dd.qpos[:3] = pos[i]
  w, x, y, zz = quat[i]
  dd.qpos[3:7] = [w, -x, -y, -zz]  # mc_rtc logs the inverse rotation
  dd.qpos[adr] = q[i]
  mujoco.mj_kinematics(m, dd)
  for f in range(2):
    R = dd.xmat[feet[f]].reshape(3, 3)
    fx = R[:, 0].copy()
    fx[2] = 0
    fx /= np.linalg.norm(fx)
    fy = np.array([-fx[1], fx[0], 0.0])
    nrm = R[:, 2]
    tilt[k, f] = np.degrees([np.arctan2(-(nrm @ fx), nrm[2]), np.arctan2(nrm @ fy, nrm[2])])
    fyaw[k, f] = np.arctan2(fx[1], fx[0])
    fp[k, f] = dd.xpos[feet[f]]
tk = t[idx]
Fk = F[idx]
flat = (tk > ta - 0.5) & (tk < ta)
tilt0 = tilt[flat].mean(0)
dtilt = tilt - tilt0
qi = quat[idx]
pel_yaw = np.arctan2(2 * (-qi[:, 0] * qi[:, 3] + qi[:, 1] * qi[:, 2]), 1 - 2 * (qi[:, 2] ** 2 + qi[:, 3] ** 2))  # inverse logged
toe = np.degrees(np.angle(np.exp(1j * (fyaw - pel_yaw[:, None]))))
toe0 = toe[flat].mean(0)

BWm = float(Fk[flat].sum(1).mean())  # measured weight on the soles before arming
moving = (np.abs(vc).sum(1) > 1e-3) & armed
# Walk: moving for more than 2 s. Stops: armed, command off after walking, both soles carrying the robot.
ms = np.zeros(n)
for i in range(1, n):
  ms[i] = ms[i - 1] + dt if moving[i] else 0.0
walk = moving & (ms > 2.0)
wk = walk[idx]
stops = []
rel = np.where(moving[:-1] & ~moving[1:] & armed[1:])[0] + 1
for r in rel:
  nxt = np.where(moving[r:] | ~armed[r:])[0]
  end = t[r + nxt[0]] if len(nxt) else t[-1]
  stops.append((t[r], end))
# Jerk: leg target change over each 20 ms policy step, legs (RL_q is the network target).
k20 = max(1, int(round(0.02 / dt)))
d20 = np.degrees(np.abs(qt[k20::k20, LEGS] - qt[:-k20:k20, LEGS]))
dqw = d20[walk[k20::k20]]
sm = max(1, int(round(0.02 / dt)))
kern = np.ones(sm) / sm
qs = np.stack([np.convolve(q[:, j], kern, mode="same") for j in LEGS], 1)
qdd = np.gradient(np.gradient(qs, t, axis=0), t, axis=0)
qdd_rms = float(np.sqrt((qdd[walk] ** 2).mean()))

# Touchdowns: sole load rising through 30 N after 100 ms unloaded, tilt sampled 20 ms later.
lands, toes = [], []
for f in range(2):
  c = Fk[:, f] > 30
  rise = np.where(c[1:] & ~c[:-1])[0] + 1
  for r in rise:
    if not wk[r] or r < 20 or c[r - 20:r].any():
      continue
    j = min(r + 4, len(tk) - 1)
    lands.append(dtilt[j, f])
    toes.append(toe[j, f] - toe0[f])
lands, toes = np.array(lands), np.array(toes)

# Slip: travel of each sole while carrying > 30 % BW, in the first 6 s of every stop; rest = 3 s to 6 s into it.
slip, rel_slip, nstop, rest = 0.0, 0.0, 0, np.zeros(len(tk), bool)
for a, b in stops:
  w6 = (tk > a) & (tk < min(b, a + 6))
  if (Fk[w6].sum(1) < 0.7 * BWm).mean() > 0.2:
    continue  # lifted or falling
  nstop += 1
  rest |= (tk > a + 3) & (tk < min(b, a + 6))
  for f in range(2):
    seg = np.where(w6 & (Fk[:, f] > 0.3 * BWm))[0]
    if len(seg) > 1:
      steps = np.diff(seg) == 1
      slip += float(np.linalg.norm(np.diff(fp[seg, f, :2], axis=0), axis=1)[steps].sum())
  both = np.where(w6 & (Fk > 0.15 * BWm).all(1))[0]  # foot-to-foot vector while both carry: blind to base drift
  if len(both) > 1:
    v = fp[both, 0, :2] - fp[both, 1, :2]
    steps = np.diff(both) == 1
    rel_slip += float(np.linalg.norm(np.diff(v, axis=0), axis=1)[steps].sum())

print(f"{label}  (weight on soles before arming {BWm:.0f} N, {nstop} stop(s), base {'FloatingBase' if 'FloatingBase_position_x' in z and np.ptp(z['FloatingBase_position_x']) > 0.05 else 'MCWaiko'})")
if dqw.size:
  print(f"  jerk : leg target change per 20 ms step mean {dqw.mean():.2f} p99 {np.percentile(dqw, 99):.2f} deg | leg qdd rms {qdd_rms:.1f} rad/s^2")
if len(lands):
  print(f"  land : sole pitch med |.| {np.median(np.abs(lands[:, 0])):.1f} p90 {np.percentile(np.abs(lands[:, 0]), 90):.1f} (mean {lands[:, 0].mean():+.1f}, + toe down)"
        f" | roll med |.| {np.median(np.abs(lands[:, 1])):.1f} p90 {np.percentile(np.abs(lands[:, 1]), 90):.1f} deg"
        f" | toe-out change p90 {np.percentile(np.abs(toes), 90):.1f} deg ({len(lands)} landings)")
if nstop:
  print(f"  slip : loaded soles, first 6 s of each stop {100 * slip / nstop:.1f} cm per stop (world, drifts with MCWaiko)"
        f" | foot-to-foot while both carry {100 * rel_slip / nstop:.1f} cm per stop")
if rest.any():
  rt = dtilt[rest].mean(0)
  rl = Fk[rest].mean(0)
  print(f"  rest : sole tilt vs flat L pitch {rt[0, 0]:+.1f} roll {rt[0, 1]:+.1f} | R pitch {rt[1, 0]:+.1f} roll {rt[1, 1]:+.1f} deg"
        f" | load L/R {rl[0]:.0f}/{rl[1]:.0f} N ({100 * rl[0] / max(rl.sum(), 1):.0f} % left)")
