#!/usr/bin/env bash
# Fine-tune the policy that walked on the robot, with one added pressure.
#
# Resume from 2026-08-21_11-47-31 model_2100 -- the checkpoint deployed as index
# 1, verified on hardware -- and add com_step_progress. Observations and actions
# are untouched, so the weights load; only the objective changes. That keeps
# everything that transferred and asks for one thing more.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]
do sleep 60; done
sleep 30
export RHPS1_ABLATION=p0+rand+lift+steplen
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True --video-interval 12000 \
  --agent.resume True \
  --agent.load-run 2026-08-21_11-47-31 \
  --agent.load-checkpoint model_2100.pt
