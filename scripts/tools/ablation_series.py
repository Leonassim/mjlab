"""Run the policy 0 ablation ladder, one rung at a time.

Each rung trains from scratch to TARGET_IT and is then stopped: policy 0 (run
2026-07-10_20-59-17, the run whose ONNX is the deployed one) reached
track_linear_velocity 2.38 by iteration 600 and 3.09 later, while the runs
carrying every deviation plateau at 1.95. The discriminator lands within the
hour rather than overnight.

  uv run python scripts/tools/ablation_series.py                  # full ladder
  uv run python scripts/tools/ablation_series.py p0 p0+feet       # a subset

Results are appended to logs/ablation_results.md after every rung, so an
interrupted series still leaves everything it measured.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LADDER = ["p0", "p0+obs246", "p0+knee", "p0+mirror", "p0+rand", "p0+feet", "p0+prox", "p0+angmom"]
TARGET_IT = 700
WALL_LIMIT_S = 110 * 60
LOG_ROOT = Path("logs/rsl_rl/rhps1_velocity")
RESULTS = Path("logs/ablation_results.md")
TASK = "Mjlab-Velocity-Flat-RHPS1"

# 4 = both feet planted; policy 0 sat at 2.33 by iteration 600.
METRICS = [
  "Metrics/stance_contacts_mean",
  "Metrics/progress_ratio",
  "Metrics/sole_height_p90",
  "Episode_Reward/track_linear_velocity",
  "Train/mean_episode_length",
  "Policy/mean_std",
]


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
  if ea is None or "Metrics/stance_contacts_mean" not in ea.Tags()["scalars"]:
    return -1
  return ea.Scalars("Metrics/stance_contacts_mean")[-1].step


def _read(run: Path, it: int) -> dict[str, float]:
  ea = _events(run)
  out: dict[str, float] = {}
  if ea is None:
    return out
  tags = ea.Tags()["scalars"]
  for m in METRICS:
    if m not in tags:
      continue
    vals = ea.Scalars(m)
    near = min(vals, key=lambda s: abs(s.step - it))
    if abs(near.step - it) < 120:
      out[m.split("/")[-1]] = near.value
  return out


def _stop(proc: subprocess.Popen) -> None:
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
      [".venv/bin/train", TASK, "--env.scene.num-envs", "4096", "--video", "True"],
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
        status = "creche" if run_dir is None or _last_step(run_dir) < TARGET_IT else "ok"
        break
      if run_dir is not None and _last_step(run_dir) >= TARGET_IT:
        break
      if time.time() - started > WALL_LIMIT_S:
        status = "timeout"
        break
      time.sleep(20)
  finally:
    _stop(proc)

  res = {"ablation": name, "status": status, "run": run_dir.name if run_dir else "-",
         "minutes": round((time.time() - started) / 60, 1)}
  if run_dir is not None:
    res.update({k: round(v, 4) for k, v in _read(run_dir, TARGET_IT).items()})
  return res


def write_results(rows: list[dict]) -> None:
  cols = ["ablation", "status", "stance_contacts_mean", "progress_ratio", "sole_height_p90",
          "track_linear_velocity", "mean_episode_length", "mean_std", "minutes", "run"]
  lines = [f"# Ablation depuis policy 0 -- mesure a l'iteration {TARGET_IT}", "",
           "Policy 0 = run 2026-07-10_20-59-17 (son ONNX est celui deploye, md5 identique).",
           "Ses reperes : it600 track_linear_velocity 2.38 / stance 2.80 ;",
           "regime etabli 3.09 / 3.06. Les runs qui echouent plafonnent a 1.95 / 3.6-3.8.", "",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
  for r in rows:
    lines.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
  RESULTS.write_text("\n".join(lines) + "\n")


def main() -> None:
  ladder = sys.argv[1:] or LADDER
  rows: list[dict] = []
  for name in ladder:
    print(f"=== {name}", flush=True)
    row = run_rung(name)
    rows.append(row)
    write_results(rows)
    print(json.dumps(row), flush=True)
    time.sleep(30)  # let the GPU drain before the next rung


if __name__ == "__main__":
  main()
