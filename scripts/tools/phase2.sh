#!/usr/bin/env bash
# Phase 2 -- lever de pied, en reprise de la phase 1.
#
# UNE deviation : le palier `swt`, qui ramene la cible de foot_swing_height de
# 0.15 m a 0.05 m. Poids inchange a -5.0.
#
# Pourquoi celui-la et pas `soleclear` comme au plan : soleclear exige le
# palier `dense`, qui change en meme temps la forme de air_time, releve deux
# plafonds et ajoute un terme de hauteur positif -- quatre changements, pas un.
# Et le budget de Base0 designe une cible plus simple. Les trois termes de
# hauteur y coutent -0.86/s a eux seuls :
#   foot_swing_height  -0.366/s   cible 0.15 m
#   min_foot_height    -0.310/s   cible 0.08 m
#   foot_clearance     -0.188/s   cible 0.15 m
# alors que le pic mesure vaut 0.0017 m. Aux trois cibles, l'erreur relative au
# carre est epinglee a ~0.98 : ce ne sont pas des objectifs, ce sont des taxes
# constantes sur l'atterrissage. Le gradient d(cout)/d(pic) vaut -13.2/m a
# 0.15 et -38.6/m a 0.05, soit 2.9x plus fort la ou la demarche vit vraiment.
#
# PORTE, 533 iterations apres la reprise :
#   peak_height_mean    > 0.004    phase 1 : ~0.0017
#   sole_clearance_p90  > 0.009    phase 1 : ~0.0065
#   fell_down          <= 0.02
#   torque_ratio       <= 0.36
# Si le pic ne bouge pas malgre 2.9x de gradient, la cause n'est pas la cible :
# passer a `fclr`, qui retire foot_clearance -- ce terme facture
# |z - 0.15| x vitesse horizontale, donc a cible lointaine il taxe le fait
# d'avancer le pied, pas le fait de le lever.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
PREV=${1:?run de la phase 1 attendue}
CKPT=${2:?checkpoint attendu}
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+cbal+swt"
export RHPS1_W_CBAL=0.5
export RHPS1_SWT_TARGET=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 1200 \
  --agent.resume True \
  --agent.load-run "$PREV" \
  --agent.load-checkpoint "$CKPT" \
  >> logs/probes/phase2.train.log 2>&1
