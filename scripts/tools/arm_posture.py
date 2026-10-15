"""arm_posture block for a NewRLQPController slot, from an mc_mujoco stand log.

  python3 scripts/tools/arm_posture.py <log.bin>

System python (mc_log_ui). The policy's own standing fixed point: mean of the
network target NewRLQPController_RL_q (controller order, legs first) 3.5-5.5 s
after arming at command 0. Also prints the first-inference jump of the target,
which is what a second pass with the measured arm_posture must bring to ~0.
"""

import sys

import mc_log_ui
import numpy as np

ORDER = ("L_CROTCH_Y L_CROTCH_R L_CROTCH_P L_KNEE_P L_ANKLE_R L_ANKLE_P CHEST_Y CHEST_P "
         "R_CROTCH_Y R_CROTCH_R R_CROTCH_P R_KNEE_P R_ANKLE_R R_ANKLE_P HEAD_Y HEAD_P "
         "L_SHOULDER_P L_SHOULDER_R L_SHOULDER_Y L_ELBOW_P L_ELBOW_Y L_WRIST_R L_WRIST_Y "
         "R_SHOULDER_P R_SHOULDER_R R_SHOULDER_Y R_ELBOW_P R_ELBOW_Y R_WRIST_R R_WRIST_Y").split()

d = mc_log_ui.read_log(sys.argv[1])
t = np.asarray(d["t"])
A = np.column_stack([d[f"NewRLQPController_RL_currentAction_{i}"] for i in range(30)])
q = np.column_stack([d[f"NewRLQPController_RL_q_{i}"] for i in range(30)])
arm = int(np.where(np.abs(np.diff(A, axis=0)).sum(1) > 1e-9)[0][0] + 1)
w = (t >= t[arm] + 3.5) & (t <= t[arm] + 5.5)
mean, std = q[w].mean(0), q[w].std(0)
jump = np.degrees(np.abs(q[arm] - q[arm - 1]))
print(f"# armed t {t[arm]:.2f}; first-inference jump max {jump.max():.3f} deg ({ORDER[int(jump.argmax())]}); "
      f"target std 3.5-5.5 s max {np.degrees(std.max()):.3f} deg")
print("    arm_posture:")
for n, v in zip(ORDER, mean):
  print(f"      {n}: {v:.5f}")
