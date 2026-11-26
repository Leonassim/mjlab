"""rest_com.py <log> <label>: CoM over the feet (heading frame) before arming, pre-walk and at rest after walking."""
import os, subprocess, sys, tempfile
import mujoco, numpy as np
_READ = """
import sys, numpy as np, mc_log_ui
l = mc_log_ui.read_log(sys.argv[1])
keep = ("Observers_Pipeline_MCWaiko_FloatingBase_world_pose_", "qIn_", "NewRLQPController_velCmd_", "NewRLQPController_RL_currentAction_", "LeftFootForceSensor_fz", "RightFootForceSensor_fz")
np.savez(sys.argv[2], **{k: np.asarray(v, float) for k, v in l.items() if k == "t" or k.startswith(keep)})
"""
ROBOT = ["L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P", "CHEST_Y", "CHEST_P",
         "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P", "HEAD_Y", "HEAD_P"]
m = mujoco.MjModel.from_xml_path("/home/lmoussafir/mjlab-rhps1/src/mjlab/asset_zoo/robots/RHPS1/xmls/RHPS1main.xml")
dd = mujoco.MjData(m)
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ROBOT]
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
with tempfile.TemporaryDirectory() as d:
  out = os.path.join(d, "l.npz")
  subprocess.run(["/usr/bin/python3", "-c", _READ, sys.argv[1], out], check=True, stderr=subprocess.DEVNULL)
  z = dict(np.load(out))
t = z["t"]; W = "Observers_Pipeline_MCWaiko_FloatingBase_world_pose_"
pos = np.stack([z[f"{W}t{a}"] for a in "xyz"], 1); quat = np.stack([z[f"{W}q{a}"] for a in "wxyz"], 1)
q = np.stack([z[f"qIn_{i}"] for i in range(16)], 1)
act = np.stack([z[f"NewRLQPController_RL_currentAction_{i}"] for i in range(30)], 1); armed = np.abs(act).sum(1) > 0
vc = z["NewRLQPController_velCmd_x"]; on = np.where(np.abs(vc) > 1e-3)[0]; t0, t1 = t[on[0]], t[on[-1]]; ta = t[armed][0]
def com_off(i):
  dd.qpos[:3] = pos[i]; w, x, y, zz = quat[i]; dd.qpos[3:7] = [w, -x, -y, -zz]; dd.qpos[adr] = q[i]
  mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
  c = dd.subtree_com[0]; f = dd.xpos[feet]; mid = f.mean(0); fw = dd.xmat[1].reshape(3, 3)[:, 0]; fw[2] = 0; fw /= np.linalg.norm(fw)
  return float(np.dot(c[:2] - mid[:2], fw[:2])), float(c[2] - mid[2])
F = np.stack([z["LeftFootForceSensor_fz"], z["RightFootForceSensor_fz"]], 1)
for name, msk in (("before arming", (t > ta - 0.5) & (t < ta)), ("pre-walk", (t > t0 - 1.5) & (t < t0 - 0.2)), ("rest after", (t > t1 + 3) & (t < t1 + 6))):
  idx = np.where(msk)[0][::10]
  r = np.array([com_off(i) for i in idx])
  print(f"{sys.argv[2]:12s} {name:14s} CoM ahead of feet mid {100 * r[:, 0].mean():+5.1f} cm, height {100 * r[:, 1].mean():5.1f} cm | Fz L/R {F[msk, 0].mean():5.0f}/{F[msk, 1].mean():5.0f} N")
