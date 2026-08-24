"""Watchdog for the gait runs. One line per event; exits when it needs a decision.

Three levels, so a run can be corrected before it has to be thrown away:

  OK     periodic progress, every PROGRESS_EVERY polls
  WARN   a soft ceiling crossed, or a metric climbing fast -- keeps running,
         this is the window for a small correction
  TRIP   a hard ceiling crossed twice -- kills the trainer and exits

The soft/hard pair is the whole point. The previous version had hard ceilings
only, so the first thing it ever said was "too late": clipping went 0.178 ->
0.276 and impact 0.131 -> 0.169 across two runs with nothing said in between,
and both had to be discarded rather than nudged.

Guards are on the deployment criteria, not on the objective. A run is allowed
to be slow at lifting its feet; it is not allowed to clip torque, land hard,
fall, or thrash the upper body, because those are what break the robot or the
transfer. Never `pgrep -f`: the pattern matches this process's own cmdline.
"""

import glob
import os
import subprocess
import sys
import time
from collections import deque

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = sys.argv[1]
POLL = float(os.environ.get("POLL", "600"))
MILESTONE = float(os.environ.get("MILESTONE", "1e9"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "3"))
RISE_FRAC = float(os.environ.get("RISE_FRAC", "0.15"))
# Polls to ignore before guarding. A resume restarts every episode at once, so
# the first samples read the reset transient, not the policy: 20 iterations in,
# this run showed jerk 1.55 and falls 0.22 against steady-state 1.00 and 0.03.
# Guarding through that kills a healthy run on its first poll.
WARMUP = int(os.environ.get("WARMUP", "3"))

# name: (tag, soft, hard, direction)   direction +1 = higher is worse
GUARDS = {
  "sat": ("Metrics/torque_saturated_frac", 0.20, 0.24, +1),
  "torque": ("Metrics/torque_limit_ratio_mean", 0.50, 0.55, +1),
  "impact": ("Metrics/pre_contact_peak_vel_mean", 0.16, 0.18, +1),
  "fall": ("Episode_Termination/fell_down", 0.05, 0.10, +1),
  "jerk": ("Episode_Metrics/mean_action_acc", 1.00, 1.20, +1),
  "upper": ("Episode_Reward/upper_body_vel_l2", -0.50, -0.80, -1),
}
# Reported, never guarded: this is the objective, and a run is allowed to be
# bad at it while it learns.
REPORT = {
  "clear": "Metrics/sole_clearance_p90",
  "period": "Metrics/step_period_mean",
  "air": "Metrics/air_time_mean",
  "len": "Metrics/step_length_mean",
  "cheat": "Metrics/sole_height_overstated_mean",
}


def emit(s):
  print(s, flush=True)


def trainer_pid():
  out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
  for line in out.splitlines():
    if "venv/bin/train" in line and "watch_gait" not in line and "ps -eo" not in line:
      return int(line.split()[0])
  return None


def sample():
  ev = sorted(glob.glob(f"{RUN}/events*"))
  if not ev:
    return None
  ea = EventAccumulator(ev[-1], size_guidance={"scalars": 4000})
  ea.Reload()
  have = ea.Tags()["scalars"]
  out, it = {}, 0
  pairs = [(k, v[0]) for k, v in GUARDS.items()] + list(REPORT.items())
  for name, tag in pairs:
    if tag not in have:
      continue
    s = ea.Scalars(tag)[-5:]
    it = max(it, s[-1].step)
    out[name] = sum(x.value for x in s) / len(s)
  out["_it"] = it
  return out


def line(s, keys):
  return " ".join(f"{k}={s[k]:.3f}" for k in keys if k in s)


hist = {k: deque(maxlen=4) for k in GUARDS}
strikes = dict.fromkeys(GUARDS, 0)
prev_it = None
stuck = 0
n = 0

pid = None
for _ in range(40):
  pid = trainer_pid()
  if pid:
    break
  time.sleep(30)
if pid is None:
  emit("DEAD trainer never appeared")
  sys.exit(4)
emit(f"watching pid {pid} run {RUN}")

while True:
  time.sleep(POLL)
  try:
    os.kill(pid, 0)
  except OSError:
    emit(f"DEAD trainer {pid} exited")
    sys.exit(2)
  s = sample()
  if not s:
    continue
  it = s["_it"]
  # Three consecutive identical iterations, not one: a poll faster than the
  # trainer's own iteration rate sees the same number twice quite legitimately.
  if it == prev_it:
    stuck += 1
    if stuck >= 3:
      emit(f"STALL iteration stuck at {it} -- wrong run dir or trainer hung")
      sys.exit(6)
  else:
    stuck = 0
  prev_it = it
  n += 1

  if n <= WARMUP:
    emit(f"warmup {n}/{WARMUP} it{it} | {line(s, REPORT)} | {line(s, GUARDS)}")
    for k in GUARDS:
      if k in s:
        hist[k].append(s[k])
    continue

  hard = []
  for k, (_, lo, hi, d) in GUARDS.items():
    if k not in s:
      continue
    v = s[k]
    hist[k].append(v)
    if (v > hi) if d > 0 else (v < hi):
      strikes[k] += 1
      if strikes[k] >= 2:
        hard.append(f"{k}={v:.3f} past {hi}")
      continue
    strikes[k] = 0
    if (v > lo) if d > 0 else (v < lo):
      emit(f"WARN it{it} {k}={v:.3f} over soft {lo} | {line(s, REPORT)}")
    elif len(hist[k]) == 4:
      old = hist[k][0]
      if old and d * (v - old) / abs(old) > RISE_FRAC:
        emit(f"WARN it{it} {k} climbing {old:.3f}->{v:.3f} over 3 polls")

  if hard:
    emit(f"TRIP it{it} " + " ".join(hard) + " -- killing trainer")
    try:
      os.kill(pid, 15)
    except OSError:
      pass
    sys.exit(3)

  if it >= MILESTONE:
    emit(f"MILESTONE it{it} | {line(s, REPORT)} | {line(s, GUARDS)}")
    sys.exit(0)

  if n % PROGRESS_EVERY == 0:
    emit(f"OK it{it} | {line(s, REPORT)} | {line(s, GUARDS)}")
