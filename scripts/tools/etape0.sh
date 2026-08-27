#!/usr/bin/env bash
# Etape 0, telle que convenue : config policy 0 + historique 5 + instrumentation
# + randomisation large + mirror loss + paires QP. AUCUN curriculum.
#
# Le curriculum vient APRES, une etape a la fois, sur une demarche etablie. Le
# tirer dans l'etape 0 a produit le sur-place de la nuit : com_step_progress
# paie clamp(duree / cible) et rien ne plafonne la periode, donc la demarche est
# partie a 1.61 s pour 3.4 cm de pas -- 2 cm/s, 77% de simple appui, suivi de
# commande a 0.61.
#
# Reprise de la premiere run a 1650 iterations, qui portait deja cette config.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+instr+rand+wide+mirror+prox"
export RHPS1_ENC_NOISE=0.005
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.resume True --agent.load-run 2026-08-26_20-16-41 \
  --agent.load-checkpoint model_1650.pt \
  --agent.max-iterations 12000 \
  >> logs/probes/etape0.train.log 2>&1
