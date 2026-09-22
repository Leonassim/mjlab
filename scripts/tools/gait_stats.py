"""Sole-level gait statistics shared by the mjlab and mc_mujoco benches."""
import numpy as np

SOLE = np.array([0.015, 0.0, -0.097])  # ankle-pitch frame; mc_mujoco sole box underside


def gait_stats(F, stance, dt, contact=None):
  """F [T, 2, 3] sole positions, stance [T, 2] bool. Returns a one-line summary."""
  v = np.linalg.norm(np.gradient(F[:, :, :2], dt, axis=0), axis=2)
  ground = np.median(F[:, :, 2][stance]) if stance.any() else F[:, :, 2].min()
  z = F[:, :, 2] - ground
  peaks = []
  for j in range(2):
    air = z[:, j] > 0.005
    d = np.diff(air.astype(int))
    ups, downs = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    for a in ups:
      b = downs[downs > a]
      if b.size and b[0] - a > int(0.1 / dt):
        peaks.append(z[a:b[0], j].max())
  T = F.shape[0] * dt
  lift = np.median(peaks) * 100 if peaks else 0.0
  sl = v[stance] * 1000 if stance.any() else np.zeros(1)
  # Double appui sur le CONTACT (force > 5 N), pas sur la hauteur de cheville :
  # la hauteur donne 40 % la ou les forces donnent 72 % sur le BWC (2026-09-19).
  c = stance if contact is None else contact
  ds = 100.0 * (c.sum(axis=1) == 2).mean()
  return (f"lever pic {lift:4.1f} cm  pas/s/pied {len(peaks) / 2 / T:4.2f}  "
          f"double appui {ds:4.1f} % (BWC 28)  "
          f"glissement appui p50 {np.median(sl):5.1f} p90 {np.percentile(sl, 90):5.1f} mm/s")
