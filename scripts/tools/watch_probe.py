"""One tab-separated sample of the metrics that decide this experiment."""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Columns 4 and 5 are read by position in watch_run.sh (torque, sat). Append,
# never reorder. peak_height_mean is gone: it resets its peak on the first
# corner touching while landing fires on all four, so it reads ~70x low.
TAGS = [
  "Metrics/step_length_mean",
  "Metrics/sole_clearance_p90",          # honest foot lift: lowest point of the sole
  "Metrics/air_time_mean",
  "Metrics/torque_limit_ratio_mean",     # col 4, guarded
  "Metrics/torque_saturated_frac",       # col 5, guarded -- clipping, not level
  "Metrics/vel_sway_rms",
  "Metrics/twist/error_vel_xy",
  "Metrics/stance_contacts_mean",
  "Episode_Termination/fell_down",
  "Episode_Metrics/mean_action_acc",     # jerk proxy
  "Metrics/step_period_mean",            # with air_time: the hover ratio
  "Metrics/pre_contact_peak_vel_mean",   # col 12, guarded -- what breaks ankles
  "Metrics/sole_height_overstated_mean", # the pitch cheat, in metres
  "Metrics/standing_single_support_rate",
  "Episode_Reward/upper_body_action_acc_l2",
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
