#!/usr/bin/env bash
# Etape 0 : nouvelle lignee. Config policy 0 + historique 5 partout +
# instrumentation + randomisation large + mirror loss + paires QP, depuis
# l'iteration 0. ~4 h.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+instr+rand+wide+mirror+prox"
export RHPS1_ENC_NOISE=0.005
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/etape0.train.log 2>&1
