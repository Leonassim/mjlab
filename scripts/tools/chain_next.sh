#!/usr/bin/env bash
# Splits flat_support into its four parameters, once the current series ends.
# fs broke at both -11 and -3.24, by opposite failure modes, so the weight is
# not the whole story: corner_tolerance is a measurement fix, change_gain is a
# new penalty mechanism, and the load/standing thresholds are a third thing.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while pgrep -f ablation_series >/dev/null; do sleep 60; done
sleep 30
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+fsct \
  p0+rand+fscg \
  p0+rand+fsload
