#!/usr/bin/env bash
# Etape 0d : la config de la policy 0, exactement, plus la randomisation de
# masse (+/-3%) et de CoM des links (+/-1 cm). Rien d'autre.
#
# Pas de mirror loss, pas d'historique sur tous les termes, pas de paires QP,
# pas de bruit encodeur reduit. L'observation reste a 126 dims, donc le format
# V3 deja en place suffit au deploiement.
#
# instr est garde : les termes de mesure a poids 1e-9 ne changent rien a
# l'entrainement (un milliardieme du budget) mais sans eux la clearance, la
# periode et les inclinaisons ne sont pas mesurees du tout -- c'est ce que
# l'etalonnage de la policy 0 a montre, ou quatre criteres lisaient nan.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+instr+masscom"
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/etape0d.train.log 2>&1
