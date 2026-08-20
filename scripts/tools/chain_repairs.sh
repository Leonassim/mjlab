#!/usr/bin/env bash
# The repair rungs, in order of expected leverage. Gated on GPU memory, never on
# pgrep -- a pattern naming the script also matches the shells watching it.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]
do sleep 60; done
sleep 30
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+swt \
  p0+rand+airtc \
  p0+rand+mfhr \
  p0+rand+fclr \
  p0+rand+airT
