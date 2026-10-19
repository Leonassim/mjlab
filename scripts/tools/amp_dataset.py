"""Build the AMP expert set from LAFAN1 walking, retargeted to RHPS1.

  uv run python scripts/tools/amp_dataset.py <out.npz> <clip.bvh> [<clip.bvh> ...]

Each clip is retargeted whole (retarget_lafan.retarget), mirrored, cut to the
frames that look like plain walking or standing at RHPS1 scale (speed below
RHPS1_AMP_VMAX m/s over 1 s, pelvis not crouched, arms hanging: elbow under 60
deg, shoulder pitch and abduction under 35 deg), runs of at least 1 s kept,
resampled to 200 Hz and turned into amp.offline_features. pair_end lists the
frames i whose (i - PAIR_STEPS, i) pair stays inside one run.
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.path.insert(0, "scripts/tools")
import retarget_lafan as rt  # noqa: E402

from mjlab.tasks.tracking.config.rhps1 import amp  # noqa: E402

VMAX = float(os.environ.get("RHPS1_AMP_VMAX", "0.8"))


def keep_mask(qs, dt, jn):
  j = {n: 7 + k for k, n in enumerate(jn)}
  n = max(1, int(1.0 / dt))
  p = qs[:, :2]
  v = np.zeros(len(qs))
  v[n // 2: len(qs) - n // 2] = np.linalg.norm(p[n:] - p[:-n], axis=-1)[: len(qs) - 2 * (n // 2)] / (n * dt)
  ok = (v < VMAX) & (qs[:, 2] > 0.78)
  for s in ("L", "R"):
    ok &= qs[:, j[f"{s}_ELBOW_P"]] > -np.radians(60)
    ok &= np.abs(qs[:, j[f"{s}_SHOULDER_P"]]) < np.radians(35)
    ok &= np.abs(qs[:, j[f"{s}_SHOULDER_R"]]) < np.radians(35)
  return ok


def runs(mask, min_len):
  out, a = [], None
  for i, k in enumerate(np.r_[mask, False]):
    if k and a is None:
      a = i
    elif not k and a is not None:
      if i - a >= min_len:
        out.append((a, i))
      a = None
  return out


def resample(qs, dt, hz=200.0):
  """qpos at hz plus world lin vel and body ang vel (central differences at the clip rate)."""
  T = len(qs)
  t = np.arange(T) * dt
  R = Rot.from_quat(qs[:, 3:7], scalar_first=True)
  lin = np.gradient(qs[:, :3], dt, axis=0)
  ang = np.zeros((T, 3))
  ang[1:-1] = (R[:-2].inv() * R[2:]).as_rotvec() / (2 * dt)
  ang[0], ang[-1] = ang[1], ang[-2]
  tn = np.arange(0, t[-1], 1.0 / hz)
  q = np.zeros((len(tn), qs.shape[1]))
  for k in [0, 1, 2] + list(range(7, qs.shape[1])):
    q[:, k] = np.interp(tn, t, qs[:, k])
  q[:, 3:7] = Slerp(t, R)(tn).as_quat(scalar_first=True)
  li = np.stack([np.interp(tn, t, lin[:, k]) for k in range(3)], -1)
  an = np.stack([np.interp(tn, t, ang[:, k]) for k in range(3)], -1)
  return q, li, an


def main():
  out, clips = sys.argv[1], sys.argv[2:]
  m = rt.model()
  jn = [m.joint(j).name for j in range(1, m.njnt)]
  feats, ends, off, total, kept = [], [], 0, 0, 0
  for c in clips:
    qs, h, *_ = rt.retarget(m, c, 0.0, 1e9)
    dt = h["dt"]
    for mirrored in (False, True):
      q = amp._mirror_qpos(qs, jn) if mirrored else qs
      mask = keep_mask(q, dt, jn)
      total += len(q)
      for a, b in runs(mask, int(1.0 / dt)):
        kept += b - a
        q2, li, an = resample(q[a:b], dt)
        f = amp.offline_features(m, q2, li, an)
        feats.append(f)
        ends.append(off + np.arange(amp.PAIR_STEPS, len(f)))
        off += len(f)
    print(f"{os.path.basename(c)}: {len(qs) * dt:.0f} s, gardes {mask.mean() * 100:.0f} % (miroir)", flush=True)
  F = np.concatenate(feats).astype(np.float32)
  np.savez_compressed(out, features=F, pair_end=np.concatenate(ends), clips=np.array(clips),
                      joint_names=np.array(jn))
  print(f"{F.shape[0]} trames a 200 Hz ({F.shape[0] / 200 / 60:.1f} min), gardees {kept / total * 100:.0f} % des trames source")


if __name__ == "__main__":
  main()
