#!/usr/bin/env bash
# Phase 6 -- liberer la cadence en desserrant le suivi de vitesse.
#
# Idee de Leo : arreter de payer le vecteur vitesse exact pour laisser le robot
# aller a son rythme pendant le pas.
#
# Deviation : `freevel`. track_linear_velocity note exp(-|c-v|^2/0.20^2) sur la
# vitesse INSTANTANEE. Un grand pas est oscillatoire par construction --
# deceleration au poser du talon, acceleration a la poussee -- donc sous un
# noyau aussi etroit chaque excursion est une erreur. Le trainement n'est pas
# ce que la politique a choisi, c'est ce que le noyau selectionne. freevel
# passe l'essentiel du poids sur direction_progress, qui filtre la vitesse sur
# un cycle avant de la noter, et elargit le noyau restant de 0.20 a 0.70.
#
# Pourquoi ca et pas un poids plus fort sur slowstep. Le calcul donne 1.30/s a
# la periode actuelle contre 1.92/s a la cible : 0.6/s pour ralentir de moitie,
# contre 2.9/s de suivi de vitesse qui s'y oppose. Monter slowstep, c'est
# pousser plus fort contre un mur ; freevel enleve le mur.
#
# Mesure des deux essais slowstep, depuis ph3 model_3750 :
#   distance 0.030 (sous la mesure)  periode 0.204 -> 0.164   PIRE
#   distance 0.050 (au-dessus)       periode 0.204 -> 0.181   toujours pire
#
# PORTE, 533 iterations apres la reprise :
#   step_period    > 0.24    ph3 : 0.204
#   double_support > 0.15    ph3 : 0.130
#   chutes        <= 0.05    couple <= 0.38
#   track_lin      > 2.0     desserre, mais la commande doit rester suivie
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+freevel"
export RHPS1_SWT_TARGET=0.05



export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 1500 \
  --agent.resume True \
  --agent.load-run 2026-08-28_03-09-45 \
  --agent.load-checkpoint model_3750.pt \
  >> logs/probes/phase7.train.log 2>&1
