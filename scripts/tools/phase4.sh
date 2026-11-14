#!/usr/bin/env bash
# Phase 4 -- consolidation. AUCUNE deviation : meme config que la phase 3,
# 1200 iterations de plus depuis model_3899.
#
# Pourquoi consolider plutot qu'ajouter. L'etalonnage de la policy 0 montre que
# les seuils de lever de pied (0.030) et d'impact (0.160) ne sont atteints par
# AUCUNE des trois politiques, y compris celle qui a marche sur le robot. Ils
# ne decrivent donc pas un objectif atteignable connu, et dimensionner une
# nouvelle deviation sur eux reviendrait a viser un nombre invente.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr"
export RHPS1_SWT_TARGET=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 1200 \
  --agent.resume True \
  --agent.load-run 2026-08-28_03-09-45 \
  --agent.load-checkpoint model_3899.pt \
  >> logs/probes/phase4.train.log 2>&1
