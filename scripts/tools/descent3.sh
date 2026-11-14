#!/usr/bin/env bash
# Terme de descente plus mordant : limite 0.20 -> 0.12, poids inchange.
#
# Reprise depuis 2026-08-31_15-11-13 model_3150, l'etat VALIDE -- le seul
# checkpoint de la lignee dont le balayage deterministe existe (5 criteres sur
# 6, index 9 deploye).
#
# Pourquoi la limite et pas le poids. Le cout vaut (descente - limite)^2 : a
# limite 0.20 il ne facture que ce qui depasse 0.20 m/s, or la descente typique
# est en dessous, donc le terme dort. 0.12 le fait mordre sur la plage ou la
# descente vit reellement. Le poids reste a -1.0 pour que l'amplitude du cout
# monte progressivement au lieu d'un saut.
#
# Ce n'est PAS un cout d'atterrissage : il se paie pendant le vol, par seconde,
# donc C7 ne s'applique pas -- la version a 0.20 l'a verifie, air time revenu a
# 0.494 sans trace de planage.
#
# Ce que la run precedente a montre : laisser tourner 5000 iterations convertit
# tout le budget en hauteur et foulee, et paie avec les chutes (0.0322), les
# couples (0.0343) et la platitude (0.0536). Il faut donc que l'impact baisse
# AVANT que la clearance ne parte trop haut, d'ou une limite plus serree des le
# depart plutot qu'un rattrapage tardif.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus+descent"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=0.0
export RHPS1_W_SWINGBONUS=6.0
export RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export RHPS1_W_DESCENT=-1.0
export RHPS1_DESCENT_LIMIT=0.12
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 5000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_15-11-13 \
  --agent.load-checkpoint model_3150.pt \
  >> logs/probes/descent3.train.log 2>&1
