#!/usr/bin/env bash
# Phase 5 -- ralentir le pas. Reprise de la PHASE 3 (model_3899), pas de la
# phase 4 : celle-ci s'est effondree des la reprise, 100 iterations apres, en
# demarrant d'emblee aux plages de commande elargies que la phase 3 avait mis
# 1200 iterations a atteindre.
#
# Deviation : `steplen` + `slowstep`. Deux paliers mais une seule idee, et
# slowstep refuse de se charger sans steplen -- il rebase le curriculum
# step_target que steplen declare.
#
# Pourquoi maintenant, et pourquoi c'est le meme probleme que le lever de pied.
# La nuit a cherche la clearance sans toucher a la cadence : 4.5 -> 8.4 mm pour
# une cible de 30-50 mm. Or lever 3 cm dans un cycle de 0.20 s demande une
# vitesse verticale que le budget de couple ne fournit pas, et une politique
# coincee la s'en sort en faisant pivoter le pied au lieu de le lever. Cycle
# lent et clearance honnete sont le meme correctif.
#
# CIBLES REBASEES sur la lignee Base0. Le palier portait 0.58 s et 0.095 m,
# calibres contre une demarche mesuree a 0.43 s et 0.082 m. La notre marche a
# 0.204 s et 0.038 m : 0.58 vaudrait 2.8x la mesure, et une cible trop loin de
# la demarche est une constante, pas un objectif. L'echelon 0 se pose juste
# au-dessus de ce qui est mesure.
#   periode  0.30 s   contre 0.204 mesure   (x1.5)
#   distance 0.050 m  contre 0.038 mesure   (x1.3)
# L'objectif de Leo reste 1 s ; 0.30 est le premier barreau, pas la cible.
#
# PORTE, 533 iterations apres la reprise :
#   step_period       > 0.24     phase 3 : 0.204
#   step_length       > 0.042    phase 3 : 0.038
#   clear_p90         >= 0.008   phase 3 : 0.0084, ne doit pas regresser
#   chutes           <= 0.02     couple <= 0.36
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+steplen+slowstep"
export RHPS1_SWT_TARGET=0.05
export RHPS1_SLOW_PERIOD=0.30
export RHPS1_STEP_DIST=0.050
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 1500 \
  --agent.resume True \
  --agent.load-run 2026-08-28_03-09-45 \
  --agent.load-checkpoint model_3899.pt \
  >> logs/probes/phase5.train.log 2>&1
