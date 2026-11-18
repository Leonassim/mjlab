"""depth_real_check.py: real Azure depth frames vs the simulated student image, same scene, same pose.

Takes a recording of static scenes (ground, boxes) and:
  1. resamples each real frame into the student's image (DEPTH_W x DEPTH_H virtual pinhole, PITCH_DEG tilt,
     CONE_DEG circle, RANGE), the median range of the real points falling in each virtual pixel;
  2. fits the ground plane on the real points (RANSAC), which gives the camera height and tilt;
  3. casts the student's rays on that plane (and on BOXES, given in the ground frame) analytically;
  4. compares: range error on pixels valid in both, holes (real misses what sim sees), extras (real returns where
     sim sees nothing), error at depth edges (mixed pixels), and writes a PNG per frame (real | sim | error).

Recording (.npz), from the Jetson stream (Azure WFOV 2x2 binned):
  depth   [T, h, w] uint16, mm (0 = invalid)
  lut     [h, w, 3] float32, unit ray of each pixel in the Azure depth frame (x right, y down, z forward), e.g.
          k4a_calibration_2d_to_3d at 1 m then normalized; it avoids depending on the lens model
  stamp   [T] float64, s (optional)
  boxes   [K, 6] float32 (optional): centre x y (m, ground frame: x along the camera heading, y left), size x y z

  python depth_real_check.py rec.npz [--out dir] [--frames 0,10,20]
  python depth_real_check.py --selftest     # synthetic Azure-like frames from the analytic scene: error ~0
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

W = int(os.environ.get("RHPS1_DEPTH_W", "64"))
H = int(os.environ.get("RHPS1_DEPTH_H", "48"))
FOVY = math.radians(float(os.environ.get("RHPS1_DEPTH_FOVY", "70")))
PITCH = math.radians(float(os.environ.get("RHPS1_DEPTH_PITCH_DEG", "25")))
CONE = math.radians(float(os.environ.get("RHPS1_DEPTH_CONE_DEG", "60")))
RANGE = tuple(float(x) for x in os.environ.get("RHPS1_DEPTH_RANGE", "0.25,2.88").split(","))
EDGE = float(os.environ.get("RHPS1_DEPTH_EDGE", "0.04"))


def virtual_rays() -> np.ndarray:
  """[H*W, 3] unit rays of the student image in the Azure frame (x right, y down, z forward), as depth_student."""
  ty = math.tan(FOVY / 2)
  tx = ty * W / H
  v = np.linspace(ty, -ty, H)
  u = np.linspace(-tx, tx, W)
  gu, gv = np.meshgrid(u, v)
  d = np.stack([gu, gv, -np.ones_like(gu)], -1).reshape(-1, 3)
  d /= np.linalg.norm(d, axis=-1, keepdims=True)
  c, s = math.cos(PITCH), math.sin(PITCH)
  tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
  d = d @ tilt.T  # MuJoCo camera axes (x right, y up, -z forward)
  return d * np.array([1.0, -1.0, -1.0])  # Azure axes


def cone_mask(rays: np.ndarray) -> np.ndarray:
  return np.arccos(np.clip(rays[:, 2], -1, 1)) <= CONE


def resample(depth_mm: np.ndarray, lut: np.ndarray) -> np.ndarray:
  """Real frame -> student image [H, W] of ranges (m), 0 = no return: median over the real points per pixel."""
  rng = depth_mm.astype(np.float64) / 1000.0
  ok = rng > 0
  p = lut[ok] * (rng[ok] / lut[ok][:, 2])[:, None]  # Azure depth is z, not range
  r = np.linalg.norm(p, axis=-1)
  q = p * np.array([1.0, -1.0, -1.0])  # MuJoCo camera axes
  c, s = math.cos(PITCH), math.sin(PITCH)
  tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
  q = q @ tilt  # undo the tilt: tilt.T @ q
  front = q[:, 2] < -1e-6
  q, r = q[front], r[front]
  ty = math.tan(FOVY / 2)
  tx = ty * W / H
  u, v = q[:, 0] / -q[:, 2], q[:, 1] / -q[:, 2]
  col = np.rint((u + tx) / (2 * tx) * (W - 1)).astype(int)
  row = np.rint((ty - v) / (2 * ty) * (H - 1)).astype(int)
  ins = (col >= 0) & (col < W) & (row >= 0) & (row < H)
  idx, r = row[ins] * W + col[ins], r[ins]
  out = np.zeros(H * W)
  order = np.argsort(idx, kind="stable")
  idx, r = idx[order], r[order]
  starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
  for a, b in zip(starts, np.r_[starts[1:], len(idx)]):
    out[idx[a]] = np.median(r[a:b])
  rays = virtual_rays()
  out[~cone_mask(rays)] = 0.0
  out[(out < RANGE[0]) | (out > RANGE[1])] = 0.0
  return out.reshape(H, W)


def fit_ground(depth_mm: np.ndarray, lut: np.ndarray, iters: int = 300, tol: float = 0.015, seed: int = 0):
  """RANSAC plane on the real points: unit normal n (toward the camera) and distance h, n.p + h = 0 on the plane."""
  rng = depth_mm.astype(np.float64) / 1000.0
  ok = rng > 0
  p = lut[ok] * (rng[ok] / lut[ok][:, 2])[:, None]
  g = np.random.default_rng(seed)
  best, best_n, best_h = -1, None, None
  for _ in range(iters):
    a, b, c = p[g.choice(len(p), 3, replace=False)]
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-9:
      continue
    n /= np.linalg.norm(n)
    h = -n @ a
    if h < 0:
      n, h = -n, -h
    k = int((np.abs(p @ n + h) < tol).sum())
    if k > best:
      best, best_n, best_h = k, n, h
  inl = np.abs(p @ best_n + best_h) < tol  # refine on the inliers (least squares)
  q = p[inl]
  cen = q.mean(0)
  n = np.linalg.svd(q - cen, full_matrices=False)[2][-1]
  h = -n @ cen
  if h < 0:
    n, h = -n, -h
  return n, h


def ground_frame(n: np.ndarray, h: float):
  """Rotation R (rows: ground x forward, y left, z up, in Azure axes) and the camera height h."""
  z = n  # plane normal toward the camera = up
  fwd = np.array([0.0, 0.0, 1.0]) - z * z[2]
  x = fwd / np.linalg.norm(fwd)
  y = np.cross(z, x)
  return np.stack([x, y, z]), h


def cast(rays: np.ndarray, R: np.ndarray, h: float, boxes: np.ndarray | None) -> np.ndarray:
  """Ranges of the rays (Azure axes) on the plane z=0 and the boxes (ground frame, camera at (0, 0, h))."""
  d = rays @ R.T
  o = np.array([0.0, 0.0, h])
  t = np.full(len(d), np.inf)
  down = d[:, 2] < -1e-9
  t[down] = -h / d[down, 2]
  for bx in boxes if boxes is not None else []:
    c, sz = np.array([bx[0], bx[1], bx[5] / 2]), np.array(bx[3:6]) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
      t1 = (c - sz - o) / d
      t2 = (c + sz - o) / d
    tn = np.nanmax(np.minimum(t1, t2), -1)
    tf = np.nanmin(np.maximum(t1, t2), -1)
    hit = (tn <= tf) & (tf > 0)
    t = np.where(hit, np.minimum(t, np.maximum(tn, 0)), t)
  t[~cone_mask(rays) | (t < RANGE[0]) | (t > RANGE[1])] = 0.0
  return t.reshape(H, W)


def compare(real: np.ndarray, sim: np.ndarray) -> dict:
  both = (real > 0) & (sim > 0)
  err = real - sim
  gy, gx = np.gradient(sim)
  edge = (np.hypot(gx, gy) > EDGE) & both
  flat = both & ~edge
  sv = sim > 0
  return {
    "valid_real": float((real > 0).mean()), "valid_sim": float(sv.mean()),
    "holes": float(((real == 0) & sv).sum() / max(sv.sum(), 1)),
    "extras": float(((real > 0) & ~sv).sum() / max((~sv).sum(), 1)),
    "bias_cm": float(100 * np.median(err[flat])) if flat.any() else float("nan"),
    "mae_cm": float(100 * np.abs(err[flat]).mean()) if flat.any() else float("nan"),
    "p95_cm": float(100 * np.percentile(np.abs(err[flat]), 95)) if flat.any() else float("nan"),
    "edge_mae_cm": float(100 * np.abs(err[edge]).mean()) if edge.any() else float("nan"),
  }


def save_png(path: Path, real: np.ndarray, sim: np.ndarray) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
  m = np.ma.masked_equal
  ax[0].imshow(m(real, 0), vmin=0, vmax=RANGE[1], cmap="viridis"); ax[0].set_title("real (resampled)")
  ax[1].imshow(m(sim, 0), vmin=0, vmax=RANGE[1], cmap="viridis"); ax[1].set_title("sim (same pose)")
  e = np.where((real > 0) & (sim > 0), 100 * (real - sim), np.nan)
  im = ax[2].imshow(e, vmin=-5, vmax=5, cmap="coolwarm"); ax[2].set_title("real - sim (cm)")
  fig.colorbar(im, ax=ax[2])
  for a in ax:
    a.axis("off")
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def synthetic(T: int = 3, hw: int = 512, h: float = 1.35, tilt_deg: float = 50.0, seed: int = 0):
  """Azure-like recording of the analytic scene: equidistant fisheye, 120 deg, depth in mm, noise 6 mm."""
  j, i = np.meshgrid(np.arange(hw) + 0.5, np.arange(hw) + 0.5)
  x, y = (j - hw / 2) / (hw / 2), (i - hw / 2) / (hw / 2)
  r = np.hypot(x, y)
  th, ph = r * math.radians(60), np.arctan2(y, x)
  lut = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).astype(np.float32)
  a = math.radians(tilt_deg)  # optical axis below the horizontal
  n = np.array([0.0, -math.cos(a), -math.sin(a)])  # world up in Azure axes (x right, y down, z forward)
  R, _ = ground_frame(n, h)
  boxes = np.array([[1.0, 0.1, 0.0, 0.5, 0.4, 0.245]], dtype=np.float32)
  g = np.random.default_rng(seed)
  flat_rays = lut.reshape(-1, 3).astype(np.float64)
  d = flat_rays @ R.T
  o = np.array([0.0, 0.0, h])
  t = np.full(len(d), np.inf)
  down = d[:, 2] < -1e-9
  t[down] = -h / d[down, 2]
  for bx in boxes:
    c, sz = np.array([bx[0], bx[1], bx[5] / 2]), np.array(bx[3:6]) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
      t1, t2 = (c - sz - o) / d, (c + sz - o) / d
    tn, tf = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
    hit = (tn <= tf) & (tf > 0)
    t = np.where(hit, np.minimum(t, np.maximum(tn, 0)), t)
  z = t * flat_rays[:, 2]
  z[~np.isfinite(z) | (r.reshape(-1) > 1) | (t > 4.0)] = 0
  depth = np.stack([np.clip(np.rint(1000 * (z + (z > 0) * g.normal(0, 0.006, z.shape))), 0, 65535)
                    for _ in range(T)]).reshape(T, hw, hw).astype(np.uint16)
  return {"depth": depth, "lut": lut, "boxes": boxes}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("rec", nargs="?")
  ap.add_argument("--out", default="logs/stairs_runs/real_check")
  ap.add_argument("--frames", default="")
  ap.add_argument("--selftest", action="store_true")
  a = ap.parse_args()
  rec = synthetic() if a.selftest else dict(np.load(a.rec))
  out = Path(a.out) / ("selftest" if a.selftest else Path(a.rec).stem)
  out.mkdir(parents=True, exist_ok=True)
  depth, lut = rec["depth"], rec["lut"].astype(np.float64)
  boxes = rec.get("boxes")
  frames = [int(x) for x in a.frames.split(",")] if a.frames else range(len(depth))
  rays = virtual_rays()
  rows = []
  for k in frames:
    n, h = fit_ground(depth[k], lut)
    R, h = ground_frame(n, h)
    real = resample(depth[k], lut)
    sim = cast(rays, R, h, boxes)
    s = compare(real, sim)
    down = math.degrees(math.asin(np.clip(-n[2], -1, 1)))
    rows.append(s)
    print(f"frame {k}: cam height {h:.3f} m, optical axis {down:.1f} deg below horizontal | "
          + " ".join(f"{key} {v:.3f}" for key, v in s.items()), flush=True)
    save_png(out / f"frame_{k:04d}.png", real, sim)
  keys = rows[0].keys()
  print("MEAN " + " ".join(f"{key} {np.nanmean([r[key] for r in rows]):.3f}" for key in keys))
  print(f"PNGs in {out}")


if __name__ == "__main__":
  main()
