#!/usr/bin/env bash
# The full run: policy 0 + randomisation + the lift package.
# video-interval is in env.step() calls, and one iteration is
# num_steps_per_env = 48 of them, so 12000 is a clip every 250 iterations.
# The 2000 default is one every 42 -- 360 clips over a full run. Knee stays at
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
  --env.scene.num-envs 4096 --video True --video-interval 12000
