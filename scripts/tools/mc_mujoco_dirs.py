"""One mc_mujoco bench walk: displacement, heading, steps, falls, and QP collision dampers.

  python3 scripts/tools/mc_mujoco_dirs.py <log.bin>

System python (mc_log_ui). The heading is taken from the conjugate of the logged
FloatingBase quaternion: mc_rtc logs the inverse rotation. The QP part reads the
CollisionMonitor channels (distance, di, ds per pair) and reports the share of
the commanded window spent inside each damper zone.
"""

import sys

import mc_log_ui
import numpy as np

d = mc_log_ui.read_log(sys.argv[1])
t = np.asarray(d["t"])
dt = float(np.median(np.diff(t)))
ob = np.stack([np.asarray(d[f"NewRLQPController_RL_currentObservation_{i}"]) for i in range(126)], -1)
cmd = ob[:, 123:126]
on = np.where(np.abs(cmd).max(1) > 0.5)[0]
if len(on) == 0:
  print("commande jamais vue")
  sys.exit()
a, b = on[0], on[-1]
x, y, z = (np.asarray(d[f"FloatingBase_position_{c}"]) for c in "xyz")
qw, qx, qy, qz = (np.asarray(d[f"FloatingBase_orientation_{c}"]) for c in "wxyz")
qx, qy, qz = -qx, -qy, -qz
yaw = np.degrees(np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))))
lf, rf = np.asarray(d["LeftFootForceSensor_fz"]), np.asarray(d["RightFootForceSensor_fz"])


def steps(f):
  off = (f[a:b] < 20).astype(int)
  e = np.diff(np.r_[0, off, 0])
  n = (np.where(e == -1)[0] - np.where(e == 1)[0]) * dt
  return int((n > 0.15).sum())


fell = z[a:].min() < 0.6
print(f"cmd {np.round(cmd[b], 2)} {t[b] - t[a]:.0f}s: dx {x[b] - x[a]:+.2f} dy {y[b] - y[a]:+.2f} m, "
      f"lacet {yaw[b] - yaw[a]:+.0f} deg, pas G {steps(lf)} D {steps(rf)}, chute {'OUI' if fell else 'non'}")
pairs = sorted({k[len("CollisionMonitor_"):-len("_distance")] for k in d if k.startswith("CollisionMonitor_")
                and k.endswith("_distance")})
hits = []
for p in pairs:
  dist = np.asarray(d[f"CollisionMonitor_{p}_distance"])[a:b]
  di = np.asarray(d[f"CollisionMonitor_{p}_di"])[a:b]
  ds = np.asarray(d[f"CollisionMonitor_{p}_ds"])[a:b]
  inz = np.mean(dist < di) * 100
  if inz > 0:
    hits.append(f"{p} {inz:.1f}% (min {dist.min() * 100:.1f} cm, di {di[0] * 100:.1f}, ds {ds[0] * 100:.1f})")
print("  QP freine : " + ("; ".join(hits) if hits else "jamais"))
