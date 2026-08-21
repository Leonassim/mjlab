#!/usr/bin/env bash
# The full run: policy 0 + randomisation + the lift package. Knee stays at
# 100 N.m and torque_limit_margin is untouched -- the point is to raise the foot
# without spending the property policy 0 was kept for.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]
do sleep 30; done
sleep 20
export RHPS1_ABLATION=p0+rand+lift
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True
