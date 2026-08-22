#!/usr/bin/env bash
# Fine-tune the policy that walked on the robot, with the gait constraints lifted.
#
# Resume from 2026-08-21_11-47-31 model_2100 -- the checkpoint deployed as index
# 1, verified on hardware -- and change the objective only. Observations and
# actions are untouched, so the weights load: everything that transferred is
# kept, and three things are asked for on top.
#
#   steplen   pay per touchdown, squared, for ground covered  (adds the target)
#   freevel   direction, not the exact velocity vector        (removes a wall)
#   freeroll  roll/pitch kernel widened, yaw untouched        (removes a wall)
#
# Watch Metrics/step_length_mean against the measured 2.8 cm, and
# Metrics/vel_sway_rms -- if the weight transfer never appears, the walls were
# not what held the gait back.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]
do sleep 60; done
sleep 30
export RHPS1_ABLATION=${RHPS1_ABLATION:-p0+rand+lift+steplen+freevel+freeroll+footladder}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True --video-interval 250 \
  --agent.resume True \
  --agent.load-run ${LOAD_RUN:-2026-08-21_11-47-31} \
  --agent.load-checkpoint ${LOAD_CKPT:-model_2100.pt}
