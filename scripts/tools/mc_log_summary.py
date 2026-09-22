"""Summarise an mc_mujoco bench log. System python3 (mc_log_ui needs PyQt5).

  python3 scripts/tools/mc_log_summary.py [log.bin] [--arm 3.0]
"""
import sys
import numpy as np
import mc_log_ui

args = [x for x in sys.argv[1:] if not x.startswith("--export=")]
export = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--export=")), None)
path = args[0] if args else "/home/lmoussafir/logs/mc_rtc/mc-control-NewRLQPController-latest.bin"
d = mc_log_ui.read_log(path)
i0 = int(3.0 / 0.005)
x, y, z = (np.array(d["FloatingBase_position_" + k]) for k in "xyz")
q = {c: np.array(d["FloatingBase_orientation_" + c]) for c in "wxyz"}
yaw = np.degrees(np.unwrap(np.arctan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2))))
n = len(x); T = (n - 1 - i0) * 0.005
j12 = min(i0 + 2400, n - 1)
air = [1 - np.array(d[f"NewRLQPController_{s}FootContact"]).astype(float)[i0:].mean() for s in ("left", "right")]
nq = len([k for k in d if k.startswith("qIn_") and k[4:].isdigit()])
qin = np.array([d[f"qIn_{j}"] for j in range(nq)])[:, i0:]
print(f"mc_mujoco {T:.1f} s  12 s: dist {np.hypot(x[j12]-x[i0], y[j12]-y[i0]):.2f} m  lacet {abs(yaw[j12]-yaw[i0]):.0f} deg   "
      f"fin: dist {np.hypot(x[-1]-x[i0], y[-1]-y[i0]):.2f} m  lacet {abs(yaw[-1]-yaw[i0]):.0f} deg  "
      f"z min {z[i0:].min():.3f}  en l air {np.mean(air):.2f}  articul. max {np.degrees(np.ptp(qin, 1)).max():.1f} deg")
if export:
  A = np.array([d[f"NewRLQPController_RL_currentAction_{i}"] for i in range(30)]).T
  np.savez(export, p=np.stack([x, y, z], 1), q=np.array([d["FloatingBase_orientation_" + c] for c in "wxyz"]).T,
           qin=np.array([d[f"qIn_{j}"] for j in range(nq)]).T, armed=np.abs(A).sum(1) > 0,
           fz=np.array([d["LeftFootForceSensor_fz"], d["RightFootForceSensor_fz"]]).T)
