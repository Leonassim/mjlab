"""Three assertions that keep the ablation ladder honest.

  1. `RHPS1_ABLATION=p0` differs from policy 0's recorded env.yaml only by the
     entries in EXPLAINED, each with a reason. Anything else means the baseline
     is not policy 0 and no rung above it can be trusted.

  2. Same against agent.yaml, via EXPLAINED_AGENT.

  3. Applying every rung lands back on the untouched configuration. A deviation
     missing from all rungs would otherwise slip through the whole ladder
     untested -- which is how three runs carried forty differences at once.

Run with no arguments. Exits non-zero on either failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REF_RUN = Path("logs/rsl_rl/rhps1_velocity/2026-07-10_20-59-17/params")
REF = _REF_RUN / "env.yaml"
REF_AGENT = _REF_RUN / "agent.yaml"

# Differences between policy 0 and the `p0` rung that are representation or
# inert, each checked by hand. Prefix match on the diff path.
EXPLAINED: dict[str, str] = {
  "commands.twist.vel_ramp_rate": "field added since, None is inert",
  "metrics.command_progress": "logging only",
  "metrics.sole_height": "logging only",
  "rewards.flat_support.params.command_name": "required by the class, no July equivalent needed",
  # Not irreducible after all -- every extension flat_support_penalty grew has a
  # neutral setting, and the first p0 rung ran with corner_tolerance 0.001, i.e.
  # counting corners by height rather than by the solver's contact detection.
  "rewards.flat_support.params.corner_tolerance": "0.0 restores `contacts = found > 0`",
  "rewards.flat_support.params.change_gain": "0.0 drops the corner-loss term July had not",
  "rewards.flat_support.params.standing_threshold": "-1.0 makes `standing` never true",
  "rewards.flat_support.params.load_threshold": "0.0 collapses `loaded` to `in_contact`",
  "rewards.standing_single_support.params.grace_period": "0.0 is exactly July's formula",
  "rewards.stance_action_acc_l2.params": "index list -> asset_cfg, same 6+6 joints",
  "rewards.torque_limit_margin.params.asset_cfg": "default is every joint, inert",
  "rewards.upper_body_action_acc_l2.func": "renamed, same body",
  "rewards.upper_body_action_acc_l2.params": "index list -> asset_cfg, same 18 joints",
  "scene.entities.robot.articulation.actuators": ".*_KNEE_P split L/R, fields equal",
  "viewer.reward_bar_max_terms": "viewer only",
}


# Same for agent.yaml. All three are verified inert rather than reverted: the
# rsl_rl constructors reject unknown kwargs, so faking policy 0's plain PPO
# would mean deleting fields the current runner needs.
EXPLAINED_AGENT: dict[str, str] = {
  "actor.distribution_cfg.std_range": (
    "cap 1.3 never approached: mean_std peaks 1.001 at it 0, falls to 0.415"
  ),
  "algorithm.class_name": "TorqueGuidedPPO with coef 0 is PPO (rl_ext.py:383)",
  "algorithm.symmetry_cfg": "augmentation off, mirror loss off on this rung",
  "algorithm.torque_guidance_coef": "0.0, gated by `if coef > 0.0`",
  "algorithm.torque_guidance_obs_group": "unused at coef 0",
  "algorithm.torque_guidance_obs_term": "unused at coef 0",
  "algorithm.torque_guidance_warmup_updates": "unused at coef 0",
}


class _L(yaml.SafeLoader):
  pass


def _obj(loader, suffix, node):
  del suffix
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  return loader.construct_scalar(node)


for _tag in (
  "tag:yaml.org,2002:python/object/apply:",
  "tag:yaml.org,2002:python/object/new:",
  "tag:yaml.org,2002:python/object:",
):
  _L.add_multi_constructor(_tag, _obj)
_L.add_multi_constructor(
  "tag:yaml.org,2002:python/name:", lambda l, s, n: s.split(".")[-1]
)
_L.add_multi_constructor(
  "tag:yaml.org,2002:python/tuple", lambda l, s, n: l.construct_sequence(n, deep=True)
)

SKIP = {"num_envs", "device", "seed"}


def _walk(a, b, path="", out=None):
  out = [] if out is None else out
  if isinstance(a, dict) and isinstance(b, dict):
    for k in sorted(set(a) | set(b)):
      if k in SKIP:
        continue
      p = f"{path}.{k}" if path else k
      if k not in a:
        out.append(p)
      elif k not in b:
        out.append(p)
      else:
        _walk(a[k], b[k], p, out)
  elif isinstance(a, list) and isinstance(b, list):
    if len(a) != len(b):
      out.append(path)
    else:
      for i, (x, y) in enumerate(zip(a, b)):
        _walk(x, y, f"{path}[{i}]", out)
  elif isinstance(a, float) and isinstance(b, float):
    if abs(a - b) > 1e-9:
      out.append(path)
  elif a != b:
    out.append(path)
  return out


def _dump(ablation: str | None, agent: bool = False) -> dict:
  """Load the config in a subprocess so RHPS1_ABLATION applies at import."""
  env = dict(os.environ)
  if ablation is None:
    env.pop("RHPS1_ABLATION", None)
  else:
    env["RHPS1_ABLATION"] = ablation
  loader = "load_rl_cfg" if agent else "load_env_cfg"
  out = Path(tempfile.mkdtemp()) / ("agent.yaml" if agent else "env.yaml")
  code = (
    "from pathlib import Path\n"
    "from mjlab.utils.os import dump_yaml\n"
    f"from mjlab.tasks.registry import {loader}\n"
    f"dump_yaml(Path({str(out)!r}), {loader}('Mjlab-Velocity-Flat-RHPS1'))\n"
  )
  subprocess.run(
    [sys.executable, "-c", code], env=env, check=True, capture_output=True
  )
  return yaml.load(out.open(), Loader=_L)


def main() -> int:
  if not REF.exists():
    print(f"reference missing: {REF}", file=sys.stderr)
    return 2

  from mjlab.tasks.velocity.config.rhps1.ablation import LADDER

  ref = yaml.load(REF.open(), Loader=_L)
  failed = False

  print("1. baseline `p0` against policy 0's env.yaml")
  paths = _walk(ref, _dump("p0"))
  unexplained = [
    p for p in paths if not any(p.startswith(k) for k in EXPLAINED)
  ]
  for p in sorted(paths):
    reason = next((v for k, v in EXPLAINED.items() if p.startswith(k)), None)
    print(f"   {'ok ' if reason else 'NEW'} {p}" + (f"  -- {reason}" if reason else ""))
  if unexplained:
    print(f"   FAIL {len(unexplained)} unexplained difference(s)")
    failed = True
  else:
    print(f"   pass: {len(paths)} difference(s), all explained")

  print("\n2. baseline `p0` against policy 0's agent.yaml")
  paths = _walk(yaml.load(REF_AGENT.open(), Loader=_L), _dump("p0", agent=True))
  unexplained = [
    p for p in paths if not any(p.startswith(k) for k in EXPLAINED_AGENT)
  ]
  for p in sorted(paths):
    reason = next((v for k, v in EXPLAINED_AGENT.items() if p.startswith(k)), None)
    print(f"   {'ok ' if reason else 'NEW'} {p}" + (f"  -- {reason}" if reason else ""))
  if unexplained:
    print(f"   FAIL {len(unexplained)} unexplained difference(s)")
    failed = True
  else:
    print(f"   pass: {len(paths)} difference(s), all explained")

  full = "p0+" + "+".join(LADDER)
  print(f"\n3. full ladder `{full}` against the untouched configuration")
  paths = _walk(_dump(None), _dump(full))
  for p in sorted(paths):
    print(f"   MISSING FROM EVERY RUNG  {p}")
  if paths:
    print(f"   FAIL {len(paths)} deviation(s) no rung applies")
    failed = True
  else:
    print("   pass: the ladder reconstructs the current configuration exactly")

  return 1 if failed else 0


if __name__ == "__main__":
  raise SystemExit(main())
