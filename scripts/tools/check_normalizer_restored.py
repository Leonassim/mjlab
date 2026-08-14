"""Does the observation normalisation survive loading a checkpoint?

Training never calls runner.load(). play, any measurement bench and the ONNX
export do. If EmpiricalNormalization came back blank the policy would see
mis-scaled observations everywhere except in training.

  uv run python scripts/tools/check_normalizer_restored.py <checkpoint.pt>
"""

from __future__ import annotations

import sys

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
if CKPT is None:
  raise SystemExit(__doc__)

d = torch.load(CKPT, weights_only=False, map_location="cpu")
print(f"cles du checkpoint : {sorted(d.keys())}\n")

found = False
for key, sd in d.items():
  if not isinstance(sd, dict):
    continue
  for name, t in sd.items():
    if "obs_normalizer" not in name or not isinstance(t, torch.Tensor):
      continue
    found = True
    tf = t.float()
    if name.endswith("count"):
      print(f"{key}.{name} = {int(t)}")
    else:
      print(
        f"{key}.{name:44s} min {float(tf.min()):+9.3f} "
        f"max {float(tf.max()):+9.3f} moyenne {float(tf.mean()):+9.3f}"
      )

if not found:
  print("AUCUN buffer obs_normalizer dans le checkpoint.")
  print("La normalisation n'est donc pas restauree au chargement.")
