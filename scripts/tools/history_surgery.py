"""Widen a V3 actor (126 obs) to the full-history layout (obs_format 2), same behaviour.

  uv run python scripts/tools/history_surgery.py <src_run> <ckpt> <dst_run> [depth=10]

V3: base_lin_vel[5x3], base_ang_vel[3], projected_gravity[3], joint_pos[30], joint_vel[30],
actions[30], command[5x3]. Full history: every term at `depth`, oldest first per term
(RHPS1_FULL_HISTORY, rl_controller obs_format 2). Each old input keeps its weights at its
own time slot; every other slot starts at zero weight, so the widened actor outputs
exactly what the old one did. Normalizer stats are copied per term; the optimizer state
is dropped (Adam restarts its moments).
"""

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path("logs/rsl_rl/rhps1_tracking")
TERMS = (("base_lin_vel", 3, 5), ("base_ang_vel", 3, 1), ("projected_gravity", 3, 1), ("joint_pos", 30, 1),
         ("joint_vel", 30, 1), ("actions", 30, 1), ("command", 3, 5))


def index_map(depth):
  old, new, o, n = [], [], 0, 0
  for _, dim, h in TERMS:
    for k in range(h):  # old slot k of h, oldest first -> same age in the new depth
      slot = depth - h + k
      old += list(range(o + k * dim, o + (k + 1) * dim))
      new += list(range(n + slot * dim, n + (slot + 1) * dim))
    o += dim * h
    n += dim * depth
  return old, new, o, n


def main():
  src, ck, dst = sys.argv[1], sys.argv[2], sys.argv[3]
  depth = int(sys.argv[4]) if len(sys.argv) > 4 else 10
  d = torch.load(ROOT / src / ck, map_location="cpu", weights_only=False)
  a = d["actor_state_dict"]
  old, new, n_old, n_new = index_map(depth)
  assert a["mlp.0.weight"].shape[1] == n_old, a["mlp.0.weight"].shape
  w = a["mlp.0.weight"]
  w2 = torch.zeros(w.shape[0], n_new, dtype=w.dtype)
  w2[:, new] = w[:, old]
  a["mlp.0.weight"] = w2
  for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
    v = a[key]
    v2 = torch.zeros(1, n_new, dtype=v.dtype)
    o = n = 0
    for _, dim, h in TERMS:
      newest = v[0, o + (h - 1) * dim:o + h * dim]
      for slot in range(depth):
        v2[0, n + slot * dim:n + (slot + 1) * dim] = newest
      o += dim * h
      n += dim * depth
    v2[0, new] = v[0, old]
    a[key] = v2
  d["optimizer_state_dict"]["state"] = {}
  out = ROOT / dst
  out.mkdir(parents=True, exist_ok=True)
  for f in ("params",):
    if (ROOT / src / f).exists() and not (out / f).exists():
      shutil.copytree(ROOT / src / f, out / f)
  torch.save(d, out / ck)
  print(f"{src}/{ck} -> {dst}/{ck}: actor input {n_old} -> {n_new}, {len(old)} weights moved")


if __name__ == "__main__":
  main()
