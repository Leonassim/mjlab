#!/usr/bin/env bash
# Base : la config de 2026-08-19_15-57-07, la reconstruction de la policy 0
# (dont le commit b254486c avait ete perdu), plus trois ajouts demandes par Leo :
# historique 5 sur les sept termes, mirror loss, masse et CoM randomises.
#
# PAS de `lift` : la reference est p0 brut, avec foot_swing_height a 0.15 et
# min_foot_height a 0.08. Ces valeurs saturent leurs termes, ce qui est le
# comportement de la policy 0 et donc ce qu'on reproduit.
#
# flat_touchdown a -0.018 et non -1.8. Le poids de la policy 0 valait -1.8 quand
# flat_touchdown_penalty renvoyait `cost` ; le commit 52ee92dd du 2026-08-22 est
# passe a `cost / env.step_dt`, homogene mais x200 a poids constant. Mesure a
# l'iteration 533, meme configuration nominale :
#   19 aout (repro policy 0)    -0.032/s   ratio couple 0.347
#   21 aout (run deployee)      -0.026/s   ratio couple 0.345
#   27 aout (avant correction)  -2.512/s   ratio couple 0.675
#
# DEPLOIEMENT : 510 dims, donc obs_format 5 dans rl_controller (utils.cpp
# case 5, commit 50156ea). Le format V3 a 126 dims ne convient pas.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+instr"
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/base.train.log 2>&1
