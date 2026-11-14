#!/usr/bin/env bash
# Impact traite du bon cote : cout sur la vitesse de DESCENTE pendant le vol.
#
# Reprise depuis 2026-08-31_22-19-33 model_3300 -- l'etat SAIN d'avant softland,
# clearance 0.0350, air 0.487, steplen 0.0356. Le plafond de impact_vel y revient
# a sa valeur d'origine 0.45 : softland est retire.
#
# Changement de configuration par rapport a ce checkpoint (C4 respectee) : le
# terme `descent` est neuf.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus+descent"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=0.0
export RHPS1_W_SWINGBONUS=6.0
export RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export RHPS1_W_DESCENT=-1.0
export RHPS1_DESCENT_LIMIT=0.20
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 8000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_15-11-13 \
  --agent.load-checkpoint model_3150.pt \
  >> logs/probes/descent2.train.log 2>&1
