"""Time a rollout spends inside the deployment QP's self-collision damper zones.

  uv run python scripts/tools/qp_clearance.py <rollout.npz> [<rollout.npz> ...]

Rollouts are the npz written by gait_dynamics.py. Pairs and margins are the ones
NewRLQPController runs (mc_rhps1 minimalSelfCollisions, thigh iDist corrected to
0.025). MuJoCo's collision geoms match the QP's convexes to the millimetre on
these pairs (wrist-thigh is 1.5-2.7 cm more conservative). Exact-zero mesh
distances are an artefact and are skipped.
"""

import sys

import mujoco
import numpy as np

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g  # noqa: E402

M, D = g.M, g.D
# (body1, body2, iDist, sDist); "S" is replaced by L and R.
QP = [("CHEST_P", "S_SHOULDER_Y", 0.02, 0.001), ("CHEST_P", "S_ELBOW_Y", 0.05, 0.03),
      ("CHEST_P", "S_WRIST_Y", 0.05, 0.03), ("BODY", "S_SHOULDER_Y", 0.05, 0.03),
      ("BODY", "S_ELBOW_Y", 0.05, 0.03), ("BODY", "S_WRIST_Y", 0.05, 0.03),
      ("S_CROTCH_P", "S_WRIST_Y", 0.05, 0.02), ("L_CROTCH_P", "R_CROTCH_P", 0.025, 0.01),
      ("L_KNEE_P", "R_KNEE_P", 0.02, 0.01)]


def _geom(name):
  return g._gid("rhps1_collision_" + name + ("" if name == "BODY" else "_LINK"))


PAIRS = []
for a, b, i, s in QP:
  for side in ("LR" if "S_" in a + b else "-"):
    A, B = a.replace("S_", side + "_"), b.replace("S_", side + "_")
    if (A, B) not in [(p[0], p[1]) for p in PAIRS]:
      PAIRS.append((A, B, _geom(A), _geom(B), i, s))

for f in sys.argv[1:]:
  r = np.load(f)
  names = [str(x) for x in r["names"]]
  qadr = [M.jnt_qposadr[g._jid(x)] for x in names]
  inz = np.zeros(len(PAIRS))
  sec = np.zeros(len(PAIRS))
  tot = 0
  ft = np.zeros(6)
  for e in range(r["q"].shape[0]):
    for t in range(0, r["q"].shape[1], 10):
      D.qpos[:3] = r["pos"][e, t]
      D.qpos[3:7] = r["quat"][e, t]
      D.qpos[qadr] = r["q"][e, t]
      mujoco.mj_forward(M, D)
      tot += 1
      for k, (_, _, ga, gb, i, s) in enumerate(PAIRS):
        d = mujoco.mj_geomDistance(M, D, ga, gb, 0.1, ft)
        if d == 0.0:
          continue
        inz[k] += d < i
        sec[k] += d < s
  print(f"{f}: % of time inside the QP damper zone (and below its security distance)")
  for k, (A, B, _, _, i, s) in enumerate(PAIRS):
    if inz[k]:
      print(f"  {A:12s} {B:12s} < {i * 100:4.1f} cm : {100 * inz[k] / tot:5.1f} %   "
            f"< {s * 100:.1f} cm : {100 * sec[k] / tot:4.1f} %")
