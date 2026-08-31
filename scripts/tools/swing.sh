#!/usr/bin/env bash
# foot_swing_height corrige : division par step_dt (contrainte C5) et poids
# redimensionne -5.0 -> -1.0.
#
# Le terme etait une impulsion payee a la pose SANS division par step_dt, donc
# divisee par ~200 par le RewardManager. Le broutement de contact le masquait
# en le declenchant 200 fois plus souvent ; la porte sur le temps de vol l'a
# revele -- 0.0141/s pour 6.6 attendus, 70 fois moins que com_step_progress, et
# la clearance s'est mise a baisser pendant que la foulee montait.
#
# Ecrit en entier et non derive par sed d'un script precedent : trois bugs de
# champ non remplace ont deja coute des heures, dont sept la nuit derniere.
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
  --agent.load-run 2026-08-31_08-06-15 \
  --agent.load-checkpoint model_6300.pt \
  >> logs/probes/swing.train.log 2>&1
