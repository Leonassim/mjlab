#!/usr/bin/env bash
# Horloge a poids 1.0 au lieu de 2.0.
#
# A 2.0 elle payait 3.16/s contre 1.20/s pour le suivi de vitesse : la
# politique a arbitre en faveur du rythme et a lache la commande. Le suivi
# montait a 2.32 a l'iteration 400 puis retombait a 1.20, l'erreur passant de
# 0.316 a 0.556, pendant que la recompense d'horloge montait de 2.0 a 3.16.
#
# A 1.0 l'horloge plafonne vers 1.6/s, sous le suivi a pleine sante (2.9/s).
# Reprise AVEC changement de config, le seul type qui ait tenu sur cette
# campagne.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_CLOCK=1.0
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 6000 \
  --agent.resume True \
  --agent.load-run 2026-08-30_11-57-08 \
  --agent.load-checkpoint model_2700.pt \
  >> logs/probes/clock2.train.log 2>&1
