"""Average the weights of neighbouring checkpoints of one run.

Usage: avg_checkpoints.py <run> <out_name> <iter> [<iter> ...]
  e.g. avg_checkpoints.py 2026-10-15_00-26-50 avg345 34500 35000 35500
Writes logs/rsl_rl/rhps1_tracking/<run>/model_<out_name>.pt. Optimizer state and
iteration are taken from the first checkpoint.
"""

import sys

import torch

run, name, iters = sys.argv[1], sys.argv[2], sys.argv[3:]
R = f"logs/rsl_rl/rhps1_tracking/{run}"
cks = [torch.load(f"{R}/model_{c}.pt", map_location="cpu", weights_only=False) for c in iters]
out = dict(cks[0])
for key, v in cks[0].items():
  if isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
    out[key] = {
      k: (sum(c[key][k].float() for c in cks) / len(cks)).to(t.dtype) if t.is_floating_point() else t
      for k, t in v.items()
    }
    print("averaged", key, len(v))
torch.save(out, f"{R}/model_{name}.pt")
print("saved", f"{R}/model_{name}.pt")
