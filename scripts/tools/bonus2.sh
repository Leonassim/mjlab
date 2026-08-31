#!/usr/bin/env bash
# D1, test propre de la famille immunisee a C7 : AUCUNE penalite d'atterrissage.
#
# foot_swing_height a 0.0 -- RewardManager saute les termes a poids nul, donc il
# disparait. C'est la premiere fois qu'aucun cout n'est attache a la pose.
#
# Pourquoi : avec les deux termes ensemble, le bonus valait 0.0914/s contre
# -1.2321/s pour la penalite, soit un rapport de 13. Et le rapport EMPIRE tout
# seul -- l'erreur relative de la penalite tend vers 1 quand la hauteur tombe,
# donc elle grossit a mesure que la situation se degrade (-0.44 -> -1.23).
#
# Bonus a 6.0 : a 12 mm mesures pour 50 vises le terme vaut ~0.55/s, et 2.4/s a
# la cible. Du meme ordre que gait_phase (1.97) sans le depasser.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=0.0
export RHPS1_W_SWINGBONUS=6.0
export RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 6000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_10-24-24 \
  --agent.load-checkpoint model_6900.pt \
  >> logs/probes/bonus2.train.log 2>&1
