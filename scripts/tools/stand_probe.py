"""Does the policy stand still at zero command, or step in place?

  uv run python scripts/tools/stand_probe.py <run> <ckpt.pt> [<run> <ckpt.pt> ...]

16 envs, zero command for 20 s after arming, 30 ms delay, perturbations of
RHPS1_ROBUST active. Counts lift-offs (foot force below 20 N after being
above) per env over the last 15 s, and the base drift.
"""

import sys

import numpy as np

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g  # noqa: E402

g.CMDS = {"debout": (0, 0, 0)}
g.PER = 16

if __name__ == "__main__":
  args = sys.argv[1:]
  for run, ck in zip(args[::2], args[1::2]):
    rec, dt, _ = g.rollout(run, ck)
    f = rec["f"]  # (N, T, feet) force norm
    on = f > 20.0
    lifts = (on[:, :-1] & ~on[:, 1:]).sum(1)  # (N, feet)
    per_env = lifts.sum(1)
    drift = np.linalg.norm(rec["pos"][:, -1, :2] - rec["pos"][:, 0, :2], axis=1)
    print(f"{run} {ck}: decollements par env sur 14 s  median {np.median(per_env):.0f}  max {per_env.max()}  "
          f"envs qui pietinent {np.sum(per_env > 0)}/{len(per_env)}  derive median {100 * np.median(drift):.1f} cm  max {100 * drift.max():.1f} cm")
