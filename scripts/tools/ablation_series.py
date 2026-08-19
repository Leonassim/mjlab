"""Run the policy 0 ablation ladder, one rung at a time.

Each rung trains from scratch to TARGET_IT and is then stopped. Policy 0 (run
2026-07-10_20-59-17, whose ONNX is the deployed one) reached
track_linear_velocity 2.60 and stance_contacts_mean 2.81 by iteration 600, and
settles at 3.0 / 2.89; the runs that fail plateau at 1.95 / 3.6-3.8. So the
discriminator lands inside the hour instead of overnight.

  uv run python scripts/tools/ablation_series.py                # whole ladder
  uv run python scripts/tools/ablation_series.py p0 p0+rand     # a subset

Metrics are read at two iterations, not one: a verdict on a single milestone has
been wrong in both directions before. Results are rewritten after every rung, so
an interrupted series keeps everything it measured, and rungs already in the
file are skipped on a restart.

Only one 4096-env run fits on the 2080 Ti, so this is strictly sequential. It
stops nothing but the rung it started itself.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from mjlab.tasks.velocity.config.rhps1.ablation import LADDER as RUNGS

# Cumulative: p0, p0+rand, p0+rand+obs, ...
LADDER = ["p0"] + [
  "p0+" + "+".join(RUNGS[: i + 1]) for i in range(len(RUNGS))
]

TARGET_IT = 700
CHECK_IT = 400  # second, earlier milestone -- one point is not a verdict
WALL_LIMIT_S = 100 * 60
POLL_S = 20
LOG_ROOT = Path("logs/rsl_rl/rhps1_velocity")
RESULTS = Path("logs/ablation_results.md")
TASK = "Mjlab-Velocity-Flat-RHPS1"

# The two that decide, then behaviour, then deployment feasibility.
PRIMARY = ["Episode_Reward/track_linear_velocity", "Metrics/stance_contacts_mean"]
# Ordered by what the robot is judged on: does it walk, are the actions
# executable without the QP (policy 0's were), does the foot leave the ground,
# does it follow the command. A rung that walks but pushes demand past the
# effort limit is not an improvement -- that is the property being protected.
SECONDARY = [
  "Metrics/torque_limit_ratio_mean",
  "Metrics/torque_limit_ratio_max",
  "Metrics/sole_height_p90",
  "Metrics/peak_height_mean",
  "Metrics/air_time_mean",
  "Metrics/foot_vel_max",
  "Metrics/progress_ratio",
  "Metrics/progress_walking_frac",
  "Metrics/twist/error_vel_xy",
  "Episode_Termination/fell_down",
  "Train/mean_episode_length",
  "Policy/mean_std",
]
METRICS = PRIMARY + SECONDARY

# Policy 0's own numbers at iteration 600, from its tfevents. progress_ratio and
# sole_height_p90 did not exist in July, so those columns compare rungs only.
POLICY0_AT_600 = {"track_linear_velocity": 2.60, "stance_contacts_mean": 2.81}


def _events(run: Path):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  files = sorted(run.glob("events.out.tfevents.*"))
  if not files:
    return None
  ea = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
  ea.Reload()
  return ea


def _last_step(run: Path) -> int:
  ea = _events(run)
  if ea is None:
    return -1
  tags = ea.Tags()["scalars"]
  tag = next((t for t in PRIMARY if t in tags), None)
  return ea.Scalars(tag)[-1].step if tag else -1


def _read(run: Path, it: int) -> dict[str, float]:
  """Metrics near iteration `it`, averaged over +/-25 so one sample cannot
  drive a verdict."""
  ea = _events(run)
  out: dict[str, float] = {}
  if ea is None:
    return out
  tags = ea.Tags()["scalars"]
  for m in METRICS:
    if m not in tags:
      continue
    window = [s.value for s in ea.Scalars(m) if abs(s.step - it) <= 25]
    if window:
      out[m.split("/")[-1]] = round(sum(window) / len(window), 4)
  return out


def _stop(proc: subprocess.Popen) -> None:
  """Stop the rung this script started, and nothing else."""
  if proc.poll() is not None:
    return
  os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
  for _ in range(120):
    if proc.poll() is not None:
      return
    time.sleep(1)
  os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
  proc.wait(timeout=60)


def run_rung(name: str) -> dict:
  env = dict(os.environ, RHPS1_ABLATION=name)
  before = {p.name for p in LOG_ROOT.iterdir() if p.is_dir()}
  log = Path(f"logs/ablation_{name.replace('+', '_')}.log")
  started = time.time()
  with log.open("w") as fh:
    proc = subprocess.Popen(
      [sys.executable, ".venv/bin/train", TASK,
       "--env.scene.num-envs", "4096", "--video", "True"],
      stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
      env=env, start_new_session=True,
    )

  run_dir, status = None, "ok"
  try:
    while True:
      if run_dir is None:
        new = {p.name for p in LOG_ROOT.iterdir() if p.is_dir()} - before
        if new:
          run_dir = LOG_ROOT / sorted(new)[-1]
      if proc.poll() is not None:
        reached = _last_step(run_dir) if run_dir else -1
        status = "crashed" if reached < CHECK_IT else "exited"
        break
      if run_dir is not None and _last_step(run_dir) >= TARGET_IT:
        break
      if time.time() - started > WALL_LIMIT_S:
        status = "timeout"
        break
      time.sleep(POLL_S)
  finally:
    _stop(proc)

  res = {
    "ablation": name,
    "status": status,
    "minutes": round((time.time() - started) / 60, 1),
    "run": run_dir.name if run_dir else "-",
    "log": log.name,
  }
  if status == "crashed":
    tail = log.read_text(errors="replace").strip().splitlines()[-25:]
    err = next((l for l in reversed(tail) if re.search(r"Error|Exception", l)), "")
    res["note"] = err[:120] or "see log"
  if run_dir is not None:
    res["at400"] = _read(run_dir, CHECK_IT)
    res["at700"] = _read(run_dir, TARGET_IT)
  return res


def _verdict(row: dict) -> str:
  """Walks / broken / between, on the two discriminators at both milestones."""
  if row["status"] == "crashed":
    return "crash"
  a, b = row.get("at400", {}), row.get("at700", {})
  tlv = b.get("track_linear_velocity") or a.get("track_linear_velocity")
  st = b.get("stance_contacts_mean") or a.get("stance_contacts_mean")
  if tlv is None or st is None:
    return "?"
  if tlv >= 2.4 and st <= 3.0:
    return "walks"
  if tlv <= 2.1 or st >= 3.4:
    return "broken"
  return "between"


def write_results(rows: list[dict]) -> None:
  cols = ["ablation", "verdict", "status"] + [m.split("/")[-1] for m in METRICS]
  head = [
    "# Ablation depuis policy 0",
    "",
    f"Chaque barreau part de zero et s'arrete a l'iteration {TARGET_IT} "
    f"(~1 h). Deux mesures, {CHECK_IT} et {TARGET_IT}, moyennees sur +/-25 "
    "iterations : un seul jalon s'est deja trompe dans les deux sens.",
    "",
    "Policy 0 = run 2026-07-10_20-59-17, son ONNX est celui deploye. Ses "
    "reperes a l'iteration 600 : track_linear_velocity 2.60, "
    "stance_contacts_mean 2.81 ; regime etabli 3.0 / 2.89. Les runs qui "
    "echouent plafonnent a 1.95 et 3.6-3.8.",
    "",
    "`verdict` = walks si tlv >= 2.4 et stance <= 3.0, broken si tlv <= 2.1 "
    "ou stance >= 3.4, sinon between. Cumulatif : un barreau casse designe "
    "sa propre deviation.",
    "",
    "Cellules `400 / 700`. progress_ratio et sole_height_p90 n'existaient pas "
    "en juillet : ces colonnes ne comparent que les barreaux entre eux.",
    "",
    "| " + " | ".join(cols) + " |",
    "|" + "---|" * len(cols),
  ]
  body = []
  for r in rows:
    cells = [r["ablation"], _verdict(r), r["status"]]
    for m in METRICS:
      k = m.split("/")[-1]
      a = r.get("at400", {}).get(k)
      b = r.get("at700", {}).get(k)
      cells.append("-" if a is None and b is None
                   else f"{'-' if a is None else a} / {'-' if b is None else b}")
    body.append("| " + " | ".join(str(c) for c in cells) + " |")
  tail = ["", "## Runs", ""]
  for r in rows:
    note = f" -- {r['note']}" if r.get("note") else ""
    tail.append(f"- `{r['ablation']}` : {r['run']}, {r['minutes']} min, "
                f"`logs/{r['log']}`{note}")
  RESULTS.write_text("\n".join(head + body + tail) + "\n")


def main() -> None:
  ladder = sys.argv[1:] or LADDER
  done = RESULTS.read_text() if RESULTS.exists() else ""
  rows: list[dict] = []
  for name in ladder:
    if f"| {name} |" in done:
      print(f"=== {name}: already in {RESULTS}, skipped", flush=True)
      continue
    print(f"=== {name}", flush=True)
    row = run_rung(name)
    rows.append(row)
    write_results(rows)
    print(json.dumps(row), flush=True)
    time.sleep(30)  # let the GPU drain before the next rung


if __name__ == "__main__":
  main()
