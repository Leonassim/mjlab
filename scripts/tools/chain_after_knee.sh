#!/usr/bin/env bash
# Audit first -- it is 15 minutes and it sets swt's target -- then the repairs.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
gate() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]
  do sleep 60; done
  sleep 30
}
gate
.venv/bin/python scripts/tools/reward_audit.py > logs/reward_audit.txt 2>&1
echo "=== audit rc=$?"
gate
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+swt p0+rand+airtc p0+rand+mfhr p0+rand+fclr p0+rand+airT
