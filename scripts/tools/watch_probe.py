"""One tab-separated sample of the metrics that decide this experiment."""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
  "Metrics/step_length_mean",
  "Metrics/peak_height_mean",
  "Metrics/air_time_mean",
  "Metrics/torque_limit_ratio_mean",
  "Metrics/torque_limit_ratio_max",
  "Metrics/vel_sway_rms",
  "Metrics/twist/error_vel_xy",
  "Episode_Termination/fell_down",
  "Episode_Metrics/mean_action_acc",
]

ev = sorted(glob.glob(f"{sys.argv[1]}/events*"))
if not ev:
  sys.exit(1)
ea = EventAccumulator(ev[-1], size_guidance={"scalars": 2000})
ea.Reload()
have = ea.Tags()["scalars"]
it = 0
out = []
for t in TAGS:
  if t not in have:
    out.append("-")
    continue
  s = ea.Scalars(t)[-10:]  # smooth the last few
  it = max(it, s[-1].step)
  out.append(f"{sum(x.value for x in s) / len(s):.4f}")
print("\t".join([str(it)] + out))
