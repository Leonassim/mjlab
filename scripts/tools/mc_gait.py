"""Gait stats of an mc_mujoco bench log exported by mc_log_summary.py --export.

  uv run python scripts/tools/mc_gait.py <export.npz>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bwc_log_to_csv import MODULE_ORDER  # noqa: E402
from gait_stats import SOLE, gait_stats  # noqa: E402

z = np.load(sys.argv[1])
m = mujoco.MjModel.from_xml_path("/home/lmoussafir/install/share/mc_mujoco/RHPS1/xml/RHPS1main.xml")
d = mujoco.MjData(m)
adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODULE_ORDER]
feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
idx = np.flatnonzero(z["armed"])
F = np.zeros((len(idx), 2, 3))
for k, i in enumerate(idx):
  d.qpos[:3] = z["p"][i]
  d.qpos[3:7] = z["q"][i] * np.array([1, -1, -1, -1])  # mc_rtc logs the inverse rotation
  d.qpos[adr] = z["qin"][i]
  mujoco.mj_kinematics(m, d)
  for j, b in enumerate(feet):
    F[k, j] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ SOLE
print("mc_mujoco pieds : " + gait_stats(F, z["fz"][idx] > 150, 0.005, contact=z["fz"][idx] > 5))
