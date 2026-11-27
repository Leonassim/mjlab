"""patch_cmd_normalizer.py <src.pt> <ref.pt> <dst.pt> [actor a:b] [critic c:d]: copy the observation-normaliser
statistics (mean, var, std) of the command channels from an all-axes run into a forward-only checkpoint.

Forward-only training leaves the vy / wz command channels at std 0 (vcurpA2 3000), so rsl_rl's (x - mean) / (std +
0.01) turns a side or turn command of 1 into 100: the policy reads every non-forward command as garbage and stands
(vall froze in 250 its while its own rewards paid +35/s to walk). The running count (1.3e8 samples) keeps the stats
from catching up. Defaults: actor velocity_command history 111:126, critic command and base-velocity terms 619:626
(Copy-V3-Start obs layout, obs_format 11). Count and weights are untouched.
  .venv/bin/python scripts/tools/patch_cmd_normalizer.py src/model_3000.pt ref/model_11000.pt dst/model_3000.pt
"""

import sys

import torch

src, ref, dst = sys.argv[1:4]
ranges = {"actor_state_dict": sys.argv[4] if len(sys.argv) > 4 else "111:126",
          "critic_state_dict": sys.argv[5] if len(sys.argv) > 5 else "619:626"}
S = torch.load(src, map_location="cpu", weights_only=False)
R = torch.load(ref, map_location="cpu", weights_only=False)
for g, r in ranges.items():
  a, b = (int(x) for x in r.split(":"))
  for k in ("_mean", "_var", "_std"):
    key = f"obs_normalizer.{k}"
    before = S[g][key][0, a:b].clone()
    S[g][key][0, a:b] = R[g][key][0, a:b]
    if k == "_std":
      print(f"{g} {a}:{b} std {[round(x, 3) for x in before.tolist()]} -> {[round(x, 3) for x in S[g][key][0, a:b].tolist()]}")
torch.save(S, dst)
print("wrote", dst)
