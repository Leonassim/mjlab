#!/usr/bin/env bash
# Base0 -- le nom reste tant qu'on n'a pas une politique qui marche bien sur la
# base de la policy 0 avec un peu plus de randomisation.
#
# Config de 2026-08-19_15-57-07 (reconstruction de la policy 0 apres la perte du
# commit b254486c), plus quatre ajouts demandes par Leo :
#   hist5     historique 5 sur les sept termes d'observation (126 -> 510 dims)
#   mirror    mirror loss
#   masscom   masse et inertie +/-3%, CoM des links +/-1 cm
#   prox      paires de self-collision aux distances EXACTES du QP
#
# Les distances du QP viennent de mc_rhps1/src/rhps1.cpp:150 et valent
# desormais : cuisses 0.06, genoux 0.035 (0.02 + 1.5 cm de coque mc_rtc),
# genou/cheville 0.02, bras-torse 0.05, epaule-poitrine 0.02, epaule-corps 0.05,
# poignet-cuisse 0.05. Un capteur unique a 0.02 laissait les cuisses trois fois
# moins contraintes qu'a l'execution, ce qui est ce qui bloque le pas lateral.
#
# PAS de `lift` : la reference est p0 brut. flat_touchdown a -0.018 et non -1.8,
# voir docs/plan_nuit_2026-08-27.md section 1.
#
# DEPLOIEMENT : 510 dims -> obs_format 5 (rl_controller utils.cpp case 5).
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr"
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/base0.train.log 2>&1
