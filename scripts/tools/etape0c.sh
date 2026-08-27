#!/usr/bin/env bash
# Etape 0c : config policy 0 + historique 5 + instrumentation + mirror loss +
# paires QP + biais capteurs, masse et inertie a +/-3%.
#
# Policy 0 ne randomise pas la masse du tout. `rand` l'amenait a ~+/-5% et
# `wide` a +/-12% ; a +/-12% depuis l'iteration 0 la saturation de couple des
# jambes tenait 65-70% pendant 1500 iterations et la demarche martelait a
# 0.13 s. +/-3% est un petit pas AU-DESSUS de la policy 0, pas une reduction
# en dessous.
#
# AUCUN curriculum. Il vient apres, une etape a la fois.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+instr+rand+mass3+encnoise+mirror+prox"
export RHPS1_ENC_NOISE=0.005
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/etape0c.train.log 2>&1
