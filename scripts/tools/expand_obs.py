"""Append actor (and critic) inputs to a checkpoint without changing what the policy does.

  uv run python scripts/tools/expand_obs.py <src.pt> <dst.pt> <n_actor> <means> <stds> [<n_critic> <means> <stds>]

means/stds: comma-separated, tiled over the new inputs (e.g. one value per channel of a
history-stacked term). The new first-layer columns start at zero (the policy acts exactly
as before and learns to use them), the normalizers get the given stats for them, and
Adam's second moment takes each row's mean over the old columns (a zero one makes the
first steps ~3x the learning rate whatever the gradient). A new actor obs term also needs
its mirror rule in rl_ext.py, or the mirror loss wrecks the policy.
"""
import sys

import torch


def tiled(s: str, n: int) -> torch.Tensor:
  v = torch.tensor([float(x) for x in s.split(",")])
  assert n % len(v) == 0, (n, len(v))
  return v.repeat(n // len(v))[None]


def expand(net: dict, opt_state: dict, n: int, mean: torch.Tensor, std: torch.Tensor) -> int:
  w = net["mlp.0.weight"]
  old = w.shape[1]
  net["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], n)], 1)
  for k, v in (("_mean", mean), ("_var", std**2), ("_std", std)):
    net[f"obs_normalizer.{k}"] = torch.cat([net[f"obs_normalizer.{k}"], v], 1)
  hit = 0
  for s in opt_state.values():
    for k in ("exp_avg", "exp_avg_sq"):
      if k in s and tuple(s[k].shape) == tuple(w.shape):
        pad = s[k].mean(1, keepdim=True).expand(-1, n) if k == "exp_avg_sq" else torch.zeros(w.shape[0], n)
        s[k] = torch.cat([s[k], pad], 1)
        hit += 1
  assert hit == 2, hit
  return old


a = sys.argv[1:]
c = torch.load(a[0], map_location="cpu", weights_only=False)
st = c["optimizer_state_dict"]["state"]
n = int(a[2])
old = expand(c["actor_state_dict"], st, n, tiled(a[3], n), tiled(a[4], n))
msg = f"actor {old} -> {old + n}"
if len(a) > 5:
  nc = int(a[5])
  oldc = expand(c["critic_state_dict"], st, nc, tiled(a[6], nc), tiled(a[7], nc))
  msg += f", critic {oldc} -> {oldc + nc}"
torch.save(c, a[1])
print(f"{msg}, iter {c['iter']}: {a[1]}")
