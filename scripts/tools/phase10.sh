#!/usr/bin/env bash
# Phase 8 -- run long et PROPRE de la config qui marche.
#
# Les reprises enchainees ont montre leur cout : ph3 recuperait a 2.94 depuis
# ph2, ph4 plafonnait a 1.10 depuis ph3, et chaque maillon heritait de plages
# de commande plus larges sans reapprendre a les tenir. Ici une seule reprise,
# depuis Base0 model_2100 -- checkpoint periodique d'une run saine, curriculum
# etroit -- avec swt ET fclr d'emblee au lieu de deux phases chainees.
#
# swt+fclr sont les deux seules deviations validees de la journee :
#   clearance   0.0063 -> 0.0084   +33%, visible a l'oeil sur les videos
#   pas         0.0322 -> 0.0378   +11%
#   couples     dans le budget, chutes nulles au balayage deterministe
#
# Ecarte, mesure : cbal (fige debout), slowstep (accelere le pas), freevel seul
# (fige debout), steplen (double appui 0.116 -> 0.049), consolidation de ph4.
#
# 6000 iterations : les runs de 1200 depuis un checkpoint tardif passaient la
# moitie de leur budget a recuperer de la reprise.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift"
export RHPS1_SWT_TARGET=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 4000 \
  --agent.resume True \
  --agent.load-run 2026-08-28_17-53-20 \
  --agent.load-checkpoint model_4950.pt \
  >> logs/probes/phase10.train.log 2>&1
