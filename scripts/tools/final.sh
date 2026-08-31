#!/usr/bin/env bash
# Run PROPRE depuis zero, avec la pile complete. Test decisif de D1.
#
# Quatre reprises d'affilee depuis model_6900 ont donne des transitoires
# violents dont aucune n'est revenue : suivi 0.96 apres 400 iterations, la ou
# une reprise saine revient a 1.97 en 150. Le checkpoint est un mauvais point
# de reprise, comme l'etait model_3899.
#
# Pile : les acquis mesures de la campagne, plus le bonus de hauteur.
#   hist5+mirror+masscom+prox   sim-to-real, gratuits (decision 1)
#   swt + fclr                  lever de pied (decisions 3 et 4)
#   comshift                    transfert de poids, meilleur impact (decision 10)
#   clock                       horloge, a debloque la cadence (decision 15)
#   steplen x4                  foulee (decision 17)
#   swingbonus                  hauteur payee PAR SECONDE DE VOL, immunise a C7
#
# AUCUNE penalite d'atterrissage : foot_swing_height a 0.0. C7 s'est declenchee
# cinq fois, dont une malgre l'horloge.
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
  --agent.max-iterations 12000 \
  >> logs/probes/final.train.log 2>&1
