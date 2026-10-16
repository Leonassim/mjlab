"""How strongly a policy reacts to each observation group, on real observations.

  uv run python scripts/tools/policy_sensitivity.py <obs.npy> <policy.onnx> [<policy.onnx> ...]

obs.npy: real actor observations (N, 126), e.g. sampled from a robot log during
the 5.5 Hz standing vibration. For each input group, the target oscillation that
group's real fluctuation induces: |d target / d input| * std(input), summed over
the group, worst joint, in degrees (finite differences on the ONNX, action
scales from its metadata). Also the yaw-rate -> target gain of the hip pitch and
yaw joints, the loop measured on the robot.
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort

GROUPS = {"lin_vel(hist)": range(0, 15), "gyro": range(15, 18), "gravity": range(18, 21),
          "q": range(21, 51), "qd": range(51, 81), "last action": range(81, 111),
          "command": range(111, 126)}

obs = np.load(sys.argv[1]).astype(np.float32)[:20]
sd = np.load(sys.argv[1]).astype(np.float32).std(0) + 1e-9
for path in sys.argv[2:]:
  m = onnx.load(path)
  md = {p.key: p.value for p in m.metadata_props}
  names = md["joint_names"].split(",")
  sc = np.array([float(x) for x in md["action_scale"].split(",")])
  s = ort.InferenceSession(path)
  inp = s.get_inputs()[0].name

  def act(o):
    return np.concatenate([s.run(None, {inp: o[i:i + 1]})[0] for i in range(len(o))])

  base = act(obs)
  jac = np.zeros((obs.shape[1], len(names)))
  for i in range(obs.shape[1]):
    o = obs.copy()
    h = 1e-3 * max(1.0, float(np.abs(o[:, i]).mean()))
    o[:, i] += h
    jac[i] = ((act(o) - base) / h * sc).mean(0)
  row = []
  for g, idx in GROUPS.items():
    c = (np.abs(jac[list(idx)]) * sd[list(idx)][:, None]).sum(0)
    row.append(f"{g} {np.degrees(c.max()):.3f}")
  hips = {n: jac[17, names.index(n)] for n in ("L_CROTCH_P", "R_CROTCH_P", "L_CROTCH_Y", "R_CROTCH_Y")}
  print(f"{path.split('/')[-1]}: " + " | ".join(row))
  print("   lacet -> hanches (rad par rad/s): " + ", ".join(f"{k} {v:+.4f}" for k, v in hips.items()))
