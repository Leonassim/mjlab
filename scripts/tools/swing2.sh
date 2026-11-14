#!/usr/bin/env bash
# Retour a foot_swing_height -1.0. Le -3.0 a declenche C7 : air_time 0.49 ->
# 1.55 s (le pied ne se pose plus), chutes 0.008 -> 0.83, couple 0.41 -> 0.46,
# suivi 1.33 -> 0.87. L'horloge a 3.16/s n'a PAS suffi a rendre l'atterrissage
# obligatoire.
#
# C7 confirmee une cinquieme fois, et cette fois MALGRE une horloge : une
# penalite a l'atterrissage se paie toujours en n'atterrissant pas. Le levier
# de D1 ne peut donc pas etre un cout a la pose.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=-1.0
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
  >> logs/probes/swing2.train.log 2>&1
