#!/usr/bin/env bash
# Horloge de demarche explicite (Siekmann et al.) -- prescrit le rythme au lieu
# de le recompenser apres coup.
#
# Cinq recompenses reactives ont echoue les 29-30 aout a faire ralentir le pas :
#   slowstep   accelerait (prime par pas = plus de pas)
#   freevel    le suivi de vitesse ne tenait pas la cadence
#   comprof    deportait davantage, mais en accelerant
#   capture    figeait le robot debout (0.88 de double appui)
#   poids fort saturait sans gradient
# La docstring de gait_phase_tracking decrivait deja ce mur : ces termes ne
# recompensent qu'une demarche deja decouverte, ils ne disent pas quel rythme
# essayer.
#
# Reglages de la config complete, qui sont la cible de Leo :
#   swing_duration 0.4 s, periode 2.0 s a l'arret -> 1.1 s a pleine vitesse.
# Duree et non ratio : le temps de cycle en trop va en DOUBLE APPUI et non en
# equilibre sur une jambe.
#
# DEPART A ZERO obligatoire : l'observation de phase porte l'espace a 530 dims
# (510 + 4 canaux x 5 d'historique), aucun checkpoint existant ne s'y branche.
# DEPLOIEMENT : il faudra un nouvel obs_format cote rl_controller.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock"
export RHPS1_SWT_TARGET=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/clock.train.log 2>&1
