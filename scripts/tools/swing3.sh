#!/usr/bin/env bash
# foot_swing_height -1.0 -> -3.0. Vise D1.
#
# Apres correction de la normalisation, le terme vaut -0.4357/s contre
# +1.0373/s pour com_step_progress : la politique convertit le lever en foulee,
# clearance 0.0189 -> 0.0156 pendant que steplen monte a 0.0300. -3.0 l'amene
# vers -1.3/s, a parite.
#
# RISQUE ASSUME, contrainte C7 : toute demande faite a l'atterrissage a
# jusqu'ici ete satisfaite en atterrissant moins (cbal, capture, freevel seul).
# Difference ici : l'horloge tient la cadence et vaut 3.16/s, donc cesser
# d'atterrir coute cher pour la premiere fois.
# FIL DE SECURITE : couper si double_support > 0.20 ou periode > 1.0 s.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=-3.0
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 6000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_10-24-24 \
  --agent.load-checkpoint model_6900.pt \
  >> logs/probes/swing3.train.log 2>&1
