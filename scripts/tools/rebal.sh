#!/usr/bin/env bash
# Rediomensionner le bonus de hauteur : 6.0 -> 2.0. Vise O1, O2, O3.
#
# A 6.0 il paie 4.275/s -- le plus gros poste du budget, devant upright (2.94),
# l'horloge (2.76) et trois fois le suivi de vitesse (1.35). La politique
# sacrifie donc tout pour la hauteur, ce qui explique les deux echecs de
# balayage : clearance 5.1 cm mais chutes 0.0322, couples 0.0343, platitude
# 0.0536 et impact 0.1759.
#
# 6.0 avait ete dimensionne quand le pied levait 1.2 cm et que le terme valait
# 0.55/s. A 4.3 cm de hauteur mesuree le meme poids rapporte huit fois plus.
# C'est le piege que ce depot enregistre deja : dimensionner le PLAFOND et non
# la traction de depart. A 2.0 le terme vaut ~1.4/s, du meme ordre que
# com_shift (1.43) et le suivi (1.35).
#
# descent passe de -1.0 a -4.0 : a la limite 0.12 il ne payait que -0.0105/s,
# soit 400 fois moins que le bonus. Il etait inerte, donc les deux essais
# precedents sur sa limite ne pouvaient rien donner.
#
# Reprise depuis l'etat VALIDE 2026-08-31_15-11-13 model_3150 (5 criteres sur 6,
# index 9 deploye), pas depuis les runs degradees.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus+descent"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=0.0
export RHPS1_W_SWINGBONUS=2.0
export RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export RHPS1_W_DESCENT=-4.0
export RHPS1_DESCENT_LIMIT=0.12
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 5000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_15-11-13 \
  --agent.load-checkpoint model_3150.pt \
  >> logs/probes/rebal.train.log 2>&1
