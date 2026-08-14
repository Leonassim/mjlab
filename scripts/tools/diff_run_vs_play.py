"""Comparer la config REELLEMENT UTILISEE par un run a celle que play construit.

diff_play_cfg.py compare play=False et play=True tels que le code les produit
AUJOURD'HUI. Ce n'est pas la bonne reference : les videos viennent d'un run
passe, et le depot a pu bouger depuis. La seule reference qui ne ment pas est le
env.yaml ecrit par le run lui-meme.

  uv run python scripts/tools/diff_run_vs_play.py <run_dir_ou_env.yaml>
"""

from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

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
    if all(isinstance(x, (int, float, str, bool, type(None))) for x in obj):
      out[prefix] = list(obj)
    else:
      for i, v in enumerate(obj):
        out.update(flatten(v, f"{prefix}[{i}]"))
  else:
    out[prefix] = obj
  return out


def norm(v: Any) -> str:
  """Comparer un yaml relu a une dataclass : tuple et liste sont equivalents,
  et un float relu peut differer au dernier bit."""
  if isinstance(v, (list, tuple)):
    return "[" + ",".join(norm(x) for x in v) + "]"
  if isinstance(v, float):
    return f"{v:.9g}"
  if callable(v):
    return "<callable>"
  return repr(v)


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  p = Path(sys.argv[1])
  if p.is_dir():
    p = p / "env.yaml"

  # Le yaml du run porte des tags python ; certains ne resolvent plus (une classe
  # a bouge depuis). On les remplace par leur representation textuelle plutot que
  # d'echouer : ce qui nous interesse est la valeur, pas le type.
  class Tolerant(yaml.SafeLoader):
    pass

  def unknown(loader: yaml.Loader, tag: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
      return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
      seq = loader.construct_sequence(node, deep=True)
      # !!python/object/apply:...  -> les args sont la valeur utile
      return seq[0] if len(seq) == 1 else seq
    return loader.construct_mapping(node, deep=True)

  Tolerant.add_multi_constructor("", unknown)
  Tolerant.add_multi_constructor("tag:yaml.org,2002:python/", unknown)

  run = flatten(yaml.load(p.read_text(), Loader=Tolerant))
  play = flatten(load_env_cfg(TASK, play=True))
  train_now = flatten(load_env_cfg(TASK, play=False))

  keys = sorted(set(run) | set(play))
  MISSING = object()

  def report(other: dict[str, Any], label: str) -> None:
    diffs = [k for k in keys if norm(run.get(k, MISSING)) != norm(other.get(k, MISSING))]
    print(f"\n===== run ({p}) contre {label} : {len(diffs)} champs sur {len(keys)}")
    for k in diffs:
      a = run.get(k, MISSING)
      b = other.get(k, MISSING)
      sa = "ABSENT" if a is MISSING else norm(a)
      sb = "ABSENT" if b is MISSING else norm(b)
      if len(sa) > 80:
        sa = sa[:77] + "..."
      if len(sb) > 80:
        sb = sb[:77] + "..."
      print(f"{k}\n    run  : {sa}\n    {label[:4]} : {sb}")

  report(play, "play")
  report(train_now, "train-aujourdhui")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
