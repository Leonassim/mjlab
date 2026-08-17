"""Apply the pending reward/metric renames, at a run boundary.

Renaming a reward changes its Episode_Reward/<name> key in tensorboard, and
renaming a metric changes Metrics/<name>. Mid-run that splits every curve in two
and breaks comparison against policy 0 and earlier runs, so this is staged as a
script rather than applied in place.

  uv run python scripts/tools/rename_metrics.py --dry-run
  uv run python scripts/tools/rename_metrics.py

Reward and metric keys are matched only as QUOTED strings. A bare identifier
match is not safe here: an early version renamed the "pose" reward with a word
boundary regex and would have rewritten unrelated `pose` variables in the IK
action, the viewer and the tracking task, across 24 files. Quoting also keeps
"impact_vel" off the print_impact_vel / _debug_impact_vel_sensors debug flags.

Deliberately NOT renamed -- pose, dof_pos_limits and body_ang_vel. All three are
upstream names (mujocolab/mjlab 9a3e3fdf) defined in the shared velocity config,
so renaming them moves the G1 and Go1 keys too and puts a permanent conflict in
every upstream merge. dof_pos_limits is the only term that says "dof" and
body_ang_vel is the only one abbreviating what track_angular_velocity spells
out, but neither is worth that price. Only RHPS1-local names are renamed here.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Function identifiers. Internal, invisible in tensorboard, matched as words.
FUNCTIONS = {
  # Serves six rewards including arm_torso, shoulder_chest and wrist_thigh.
  # Nothing about it is leg specific.
  "leg_proximity_cost": "proximity_cost",
}

# Keys, matched only inside quotes. Each changes a tensorboard series.
KEYS = {
  "impact_vel": "impact_velocity",  # match track_angular_velocity, spell it out
  "left_foot_marker_speed": "left_foot_speed",  # "marker" is internal jargon
  "right_foot_marker_speed": "right_foot_speed",
}

# Reads ~70x low: it resets its peak on the first corner touching while a landing
# fires on all four. sole_height_p90 replaces it. Remove the log line and its
# computation by hand -- deleting a key is not a rename.
DROP_METRICS = {"peak_height_mean"}

TARGETS = ("src/mjlab", "scripts/tools")


def iter_files():
  for t in TARGETS:
    yield from (ROOT / t).rglob("*.py")


def main() -> int:
  dry = "--dry-run" in sys.argv
  touched = 0
  for path in iter_files():
    if path.name == "rename_metrics.py":
      continue
    src = path.read_text()
    out = src
    for old, new in FUNCTIONS.items():
      out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    for old, new in KEYS.items():
      out = re.sub(rf'(["\']){re.escape(old)}\1', rf"\g<1>{new}\g<1>", out)
    if out != src:
      touched += 1
      print(f"  {path.relative_to(ROOT)}")
      if not dry:
        path.write_text(out)

  print(f"\n{touched} files {'would change' if dry else 'changed'}")
  print("\nStill to remove by hand:")
  for m in sorted(DROP_METRICS):
    print(f"  Metrics/{m}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
