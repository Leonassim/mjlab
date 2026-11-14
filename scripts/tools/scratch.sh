#!/usr/bin/env bash
# Entrainement DEPUIS ZERO avec les quatre acquis mesures. Idee de Leo, gardee
# en reserve toute la journee et maintenant indiquee : les reprises enchainees
# se degradent, et la derniere en date s'est figee debout sur les talons.
#
# Ce qui est dedans, et pourquoi :
#   hist5 + mirror + masscom + prox   gratuits, porte de phase 0 franchie
#   swt      cible foot_swing_height 0.15 -> 0.05, +39% de lever
#   fclr     retire foot_clearance, qui taxait la vitesse horizontale du pied
#   comshift transfert de poids lateral, cible 7 cm mesuree sur le BWC ;
#            meilleur impact des politiques mesurees (0.1900 au balayage)
#
# Ce qui est dehors, tout mesure et rejete aujourd'hui : cbal (fige debout),
# slowstep (accelere le pas), freevel seul (fige debout), steplen (double appui
# divise par deux), comprof (deporte plus, mais en accelerant).
#
# 12000 iterations : depuis zero il n'y a pas de dette de reprise a payer.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift"
export RHPS1_SWT_TARGET=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/scratch.train.log 2>&1
