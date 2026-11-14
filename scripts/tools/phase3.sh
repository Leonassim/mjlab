#!/usr/bin/env bash
# Phase 3 -- lever de pied, deuxieme tentative : retirer foot_clearance.
#
# UNE deviation par rapport a la phase 2 : le palier `fclr`.
#
# Pourquoi. foot_clearance facture |z - 0.15| x vitesse horizontale du pied, a
# poids -10. Le pied plafonne a 0.0025 m, donc le premier facteur vaut 0.147 et
# ne bouge pas : le terme se reduit a 1.47 x vitesse horizontale. Nomme
# clearance, il ne mesure aucune clearance -- c'est une taxe pure sur le fait
# d'avancer le pied, donc sur la longueur de pas que Leo veut augmenter.
#
# min_foot_height est GARDE malgre une saturation comparable : sa penalite
# baisse quand le pied monte, donc son gradient pointe dans le bon sens meme
# faible. foot_clearance, lui, pointe contre le pas.
#
# Ce que la phase 2 a donne (porte a 2633, depuis Base0 model_2100) :
#   peak_height     0.0018 -> 0.0025   +39%, mais porte a 0.004 : ratee
#   clearance_p90   0.0063 -> 0.0063   immobile
#   track_lin_vel   2.996  -> 3.040    intact
#   chutes          0.000  -> 0.000    intact
#   couple          0.324  -> 0.331    intact
# Donc swt est sans danger mais insuffisant seul : le gradient est 2.9x plus
# fort et ne suffit pas a vaincre une taxe qui s'oppose au pas.
#
# PORTE, 533 iterations apres la reprise :
#   peak_height     > 0.004     phase 2 : 0.0025
#   clearance_p90   > 0.009     phase 2 : 0.0063
#   step_length     en hausse   phase 2 : ~0.03
#   chutes         <= 0.02      couple <= 0.36
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
PREV=${1:?run precedente attendue}
CKPT=${2:?checkpoint attendu}
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr"
export RHPS1_SWT_TARGET=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 1200 \
  --agent.resume True \
  --agent.load-run "$PREV" \
  --agent.load-checkpoint "$CKPT" \
  >> logs/probes/phase3.train.log 2>&1
