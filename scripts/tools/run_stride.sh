#!/usr/bin/env bash
# stride + tq + nodamp, on the p0+rand base.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION=p0+rand+stride+tq+nodamp
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True --video-interval 12000
