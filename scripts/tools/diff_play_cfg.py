"""Exhaustive diff between the training config and the play config.

Flattens both dataclasses and prints only what differs. Use it before guessing
which parameter moved.

  uv run python scripts/tools/diff_play_cfg.py
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-RHPS1"


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
  out: dict[str, Any] = {}
  if is_dataclass(obj) and not isinstance(obj, type):
    obj = asdict(obj)
  if isinstance(obj, dict):
    for k, v in obj.items():
      out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
  elif isinstance(obj, (list, tuple)):
    # Short scalar lists are values (ranges, gains); lists of objects are
    # containers to descend into.
    if all(isinstance(x, (int, float, str, bool, type(None))) for x in obj):
      out[prefix] = obj
    else:
      for i, v in enumerate(obj):
        out.update(flatten(v, f"{prefix}[{i}]"))
  else:
    out[prefix] = obj
  return out


def main() -> int:
  train = flatten(load_env_cfg(TASK, play=False))
  play = flatten(load_env_cfg(TASK, play=True))

  keys = sorted(set(train) | set(play))
  diffs = [k for k in keys if repr(train.get(k)) != repr(play.get(k))]

  MISSING = object()
  print(f"{len(diffs)} champs different sur {len(keys)}\n")
  for k in diffs:
    t = train.get(k, MISSING)
    p = play.get(k, MISSING)
    ts = "ABSENT" if t is MISSING else repr(t)
    ps = "ABSENT" if p is MISSING else repr(p)
    if len(ts) > 90:
      ts = ts[:87] + "..."
    if len(ps) > 90:
      ps = ps[:87] + "..."
    print(f"{k}\n    entrainement : {ts}\n    play         : {ps}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
