#!/usr/bin/env bash
# Gate on GPU memory, not on pgrep: a pattern like `pgrep -f ablation_series`
# also matches the monitoring shells watching the chain, and that deadlocked the
# previous handoff for 36 minutes.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1

wait_gpu() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 3000 ]
  do sleep 60; done
  sleep 30
}

# 1. What the rewards say about gaits we already know. Cheap, and it decides
#    what shape the foot terms should take before we spend hours on weights.
wait_gpu
.venv/bin/python scripts/tools/reward_audit.py > logs/reward_audit.txt 2>&1
echo "=== audit done rc=$?"

# 2. The corner_tolerance rung Leo asked about; the first attempt died on a
#    wandb init timeout before iteration 400.
wait_gpu
.venv/bin/python scripts/tools/ablation_series.py p0+rand+fsct

# 3. The knee ceiling matters for the real robot, and 700 iterations is a short
#    look: policy 0 itself needed 9900. Give it room to find a gait.
wait_gpu
RHPS1_TARGET_IT=3500 RHPS1_WALL_MIN=400 \
  RHPS1_RESULTS=logs/ablation_results_long.md \
  .venv/bin/python scripts/tools/ablation_series.py p0+rand+knee
