#!/usr/bin/env bash
# Phase 1 -- contact_balance, en reprise de Base0 (model_2100).
#
# UNE seule deviation par rapport a Base0 : le palier `cbal`, qui retire
# flat_support (-0.266/s au budget) et le remplace par contact_balance, un
# BONUS par seconde d'appui sur la repartition de force entre les 4 boites.
#
# Poids 0.5, et non davantage, pour une raison precise : la recompense vaut
# somme sur les deux pieds de (evenness * charge), donc le double appui peut
# rapporter jusqu'a deux fois le simple appui. A poids 2.0 cet ecart vaudrait
# 2.0/s et ferait le travail de la phase 3 en meme temps que celui de la phase
# 1 -- deux deviations empilees, resultat inattribuable. A 0.5 l'ecart plafonne
# a 0.5/s contre track_linear_velocity a 3.03/s : marcher reste clairement
# gagnant, et l'effet mesure est bien celui de la proprete du contact.
#
# PORTE, a l'iteration 2633 (533 apres la reprise, comme Base0) :
#   contact_evenness  > 0.55   (Base0 : 0.480)
#   track_linear_vel  > 2.90   (Base0 : 3.032)
#   fell_down        <= 0.02   (Base0 : 0.000)
#   torque_ratio     <= 0.36   (Base0 : 0.324)
# Si evenness ne bouge pas, c'est le poids qui est trop faible, pas le terme :
# relire la valeur en /s avant de toucher a autre chose.
#
# A LIRE APRES LA PREMIERE ITERATION (regle 3) : Episode_Reward/contact_balance
# doit valoir ~0.27/s (0.48 d'evenness x 1.14 pied charge x 0.5). Une valeur a
# 0.001 ou a 50 veut dire que le terme est mal branche, pas mal regle.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+cbal"
export RHPS1_W_CBAL=0.5
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 4200 \
  --agent.resume True \
  --agent.load-run 2026-08-27_20-27-05 \
  --agent.load-checkpoint model_2100.pt \
  >> logs/probes/phase1.train.log 2>&1
