#!/usr/bin/env bash
# Wait for the GPU to drain, then run the ablation ladder. Only one 4096-env job
# fits on the 2080 Ti, so this starts nothing while another holds the card.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
FREE_MIB=${FREE_MIB:-9000}
for _ in $(seq 1 720); do   # up to 6 h, 30 s apart
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ $((total - used)) -ge "$FREE_MIB" ]; then
    echo "$(date -Is) GPU free (${used}/${total} MiB used), starting ladder"
    exec .venv/bin/python scripts/tools/ablation_series.py
  fi
  sleep 30
done
echo "$(date -Is) gave up waiting for the GPU"
exit 1
