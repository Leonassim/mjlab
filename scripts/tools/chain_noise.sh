#!/usr/bin/env bash
# The audit, then the same baseline at two other seeds. Every verdict on this
# ladder so far compares one run against one run; without a spread there is no
# way to tell a 15% effect from the noise it sits in.
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
exec .venv/bin/python scripts/tools/ablation_series.py p0+rand#7 p0+rand#13
