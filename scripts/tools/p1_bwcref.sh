#!/usr/bin/env bash
# P1 : reference BWC indexee par la phase, DEPUIS ZERO.
#
# Depuis zero et non en reprise, pour deux raisons. L'espace d'observation
# change (l'horloge ajoute 4 canaux par pas d'historique), donc aucun checkpoint
# existant n'est chargeable. Et T12 a montre qu'une politique convergee
# n'absorbe pas un changement de cadence : lever effondre a 0.011, meme
# signature que T7 et T8.
#
# Cibles, mesurees sur le log BWC du 2026-09-04 (20 cycles) :
#
#                            BWC     index 11      cible
#   periode/pied          0.887 s      1.705 s     ~0.9 s
#   lever de pied          7.1 cm       4.4 cm     >= 6 cm
#   |v| a la pose      0.047-0.185  0.226-0.284    <= 0.10
#   couple genou max       81 N.m   150-155 N.m    <= 100
#   genou en appui        8.6 deg      24.5 deg    <= 12 deg
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand+bwcref"
export RHPS1_DEMAND_POWER=1.0 RHPS1_DEMAND_CAP=12.0 RHPS1_W_DEMAND=-0.03
export RHPS1_W_DESCENT=-120.0 RHPS1_DESCENT_LIMIT=0.12
export RHPS1_W_SWINGBONUS=3.0 RHPS1_SWINGBONUS_H=0.05
# Haut du corps borne : le BWC ne bouge PAS les bras en marchant (0.0 deg contre
# nos 5-9), et toute la demande hors-limite de l.index 11 vient de la -- coude et
# poignet a 8-12x leur limite, jambes a zero.
export RHPS1_W_BWCREF=${RHPS1_W_BWCREF:-12.0}
export RHPS1_UPPER_SCALE=${RHPS1_UPPER_SCALE:-0.0007}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-05_03-08-04 --agent.load-checkpoint model_3000.pt \
  --agent.max-iterations 12000
