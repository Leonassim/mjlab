#!/usr/bin/env bash
# Horloge (poids 2.0, inchange) + steplen : le rythme est tenu, il manque la
# distance.
#
# La run horloge a produit air_time 0.24 (+60%), periode 0.35 (+46%) et impact
# 0.067 -- le plus bas mesure de la campagne -- mais une longueur de pas de
# 0.96 cm contre 5.0 ailleurs. Le robot pietine : il tient le rythme et leve le
# pied, il n'avance pas. Le suivi plafonnait a 1.5 pour cette raison, et il
# REMONTAIT (1.31 -> 1.54 sur 1200 iterations), il ne s'effondrait pas.
#
# steplen avait ete rejete le 29 parce qu'il divisait le double appui par deux :
# com_step_progress est paye a la pose, donc il poussait a faire plus de pas. Ce
# risque disparait ici, c'est l'horloge qui tient le cycle et non la politique.
#
# target_distance 0.03 et non 0.05 : la mesure est 0.0096, donc 0.05 laisserait
# un rapport de 0.19 et un gradient au carre de 0.037. Regle de ce fichier --
# la cible se pose au-dessus de la mesure, mais pas cinq fois au-dessus.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 6000 \
  --agent.resume True \
  --agent.load-run 2026-08-30_17-52-22 \
  --agent.load-checkpoint model_5250.pt \
  >> logs/probes/clock6.train.log 2>&1
