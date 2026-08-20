"""What every reward term actually contributes, per run.

Weights say what a term is worth in principle. This says what it was worth in
fact: the realized Episode_Reward per step, its share of the positive and
negative budget, and the raw cost behind it (value / weight). A term at 0.001 is
asleep; a term holding a third of the penalty budget is steering the gait
whether or not anyone intended it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

L = Path("logs/rsl_rl/rhps1_velocity")


def weights(run: Path) -> dict[str, float]:
  """Weights straight from the dumped yaml -- a full load needs mjlab classes."""
  y = run / "params" / "env.yaml"
  if not y.exists():
    return {}
  lines = y.read_text().splitlines()
  i = next((k for k, l in enumerate(lines) if l.startswith("rewards:")), None)
  if i is None:
    return {}
  out, cur = {}, None
  for l in lines[i + 1:]:
    if l and not l.startswith((" ", "\t")):
      break
    m = re.match(r"^  (\w+):\s*$", l)
    if m:
      cur = m.group(1)
    elif cur:
      m2 = re.match(r"^    weight:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)", l)
      if m2:
        out[cur] = float(m2.group(1))
  return out


def realized(run: Path, it: int, half: int = 25) -> dict[str, float]:
  ea = event_accumulator.EventAccumulator(
    str(run), size_guidance={event_accumulator.SCALARS: 0})
  ea.Reload()
  out = {}
  for tag in ea.Tags()["scalars"]:
    if not tag.startswith("Episode_Reward/"):
      continue
    w = [s.value for s in ea.Scalars(tag) if abs(s.step - it) <= half]
    if w:
      out[tag.split("/", 1)[1]] = sum(w) / len(w)
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("runs", nargs="+", help="run dir names under logs/rsl_rl/rhps1_velocity")
  ap.add_argument("--it", type=int, default=700)
  args = ap.parse_args()

  cols = []
  for name in args.runs:
    run = L / name
    it = args.it
    v = realized(run, it)
    if not v:  # run shorter or longer than asked; take its last point
      ea = event_accumulator.EventAccumulator(str(run))
      ea.Reload()
      tags = [t for t in ea.Tags()["scalars"] if t.startswith("Episode_Reward/")]
      it = max(s.step for s in ea.Scalars(tags[0])) if tags else it
      v = realized(run, it)
    cols.append((name, it, v, weights(run)))

  terms = sorted({t for _, _, v, _ in cols for t in v})
  pos = [sum(x for x in v.values() if x > 0) for _, _, v, _ in cols]
  neg = [sum(-x for x in v.values() if x < 0) for _, _, v, _ in cols]

  print(f"\n{'':<28}" + "".join(f"{n[:11]:>26}" for n, _, _, _ in cols))
  print(f"{'':<28}" + "".join(f"{'it ' + str(i):>26}" for _, i, _, _ in cols))
  print(f"{'term':<28}" + "".join(f"{'weight':>9}{'value':>9}{'share':>8}"
                                  for _ in cols))
  print("-" * (28 + 26 * len(cols)))
  for t in sorted(terms, key=lambda t: -max(abs(v.get(t, 0)) for _, _, v, _ in cols)):
    row = f"{t[:27]:<28}"
    for k, (_, _, v, w) in enumerate(cols):
      val = v.get(t)
      if val is None:
        row += f"{'-':>9}{'-':>9}{'-':>8}"
        continue
      budget = pos[k] if val > 0 else neg[k]
      row += (f"{w.get(t, float('nan')):>9.2f}{val:>9.4f}"
              f"{100 * abs(val) / max(budget, 1e-9):>7.1f}%")
    print(row)
  print("-" * (28 + 26 * len(cols)))
  print(f"{'sum positive':<28}" + "".join(f"{'':>9}{p:>9.3f}{'':>8}" for p in pos))
  print(f"{'sum negative':<28}" + "".join(f"{'':>9}{-n:>9.3f}{'':>8}" for n in neg))
  print("\nshare = the term's fraction of its own sign's budget. value is per "
        "step,\nalready multiplied by the weight; cost = value / weight.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
