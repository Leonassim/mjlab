#!/usr/bin/env bash
# Queued behind the running lift job. Gate on GPU memory, never on pgrep: a
# pattern naming the script also matches the shells watching it.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]
do sleep 120; done
sleep 60
export RHPS1_ABLATION=p0+rand+stride+tq
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True --video-interval 12000
