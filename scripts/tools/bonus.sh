#!/usr/bin/env bash
# swing_height_bonus_dense : bonus de hauteur PAR SECONDE DE VOL. Vise D1.
#
# foot_swing_height reste a -1.0, sa valeur saine. Le nouveau terme s'ajoute au
# lieu de le remplacer : l'un decourage un vol trop bas a la pose, l'autre paie
# la hauteur pendant le vol, et seul le second echappe a C7.
#
# Reprise depuis 2026-08-31_10-24-24 model_6900 AVEC changement de config,
# comme l'exige C4 : les deux reprises a configuration identique du journal se
# sont effondrees, et la tentative de revenir a -1.0 vient d'en fournir une
# troisieme (air_time 5.76 s, clearance 0.0029, chutes 0.56).
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=-1.0
export RHPS1_W_SWINGBONUS=2.0
export RHPS1_SWINGBONUS_H=0.05
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
  >> logs/probes/bonus.train.log 2>&1
