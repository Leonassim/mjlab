#!/usr/bin/env bash
# Rejouer la politique qui passe les six criteres, dans mjlab.
#
# Run 2026-09-01_17-45-07 model_4500. Balayage deterministe, 1024 envs,
# 11 commandes :
#   ne jamais tomber      0.0000  (seuil 0.010)
#   lever de pied         0.0326  (seuil 0.030)
#   impact faible         0.1441  (seuil 0.160)
#   couples faisables     0.0178  (seuil 0.030)
#   couples haut du corps 0.0000  (seuil 0.030)
#   pieds a plat          0.0295  (seuil 0.050)
#
# L'ablation DOIT etre identique, sinon l'environnement reconstruit ne
# correspond pas au reseau : l'horloge de demarche fournit 20 des 530 dims
# d'observation, et sans elle le chargement echoue sur la taille.
set -u
cd "$(dirname "$0")/../.." || exit 1
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
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint checkpoints/rhps1_6sur6_it4500.pt "$@"
