#!/usr/bin/env bash
# T5 : faire payer la descente. UNE deviation depuis le checkpoint de T2.
#
# Mesure en marche a 0.2 m/s (impact_walking.py), 256 envs tous vivants :
#
#                       policy 0      6/6
#   descente p50          0.1476   0.2218   1.5x
#   descente p99          0.1859   0.3456   1.9x
#   force max (x poids)     1.82x    2.57x
#
# Le BWC atterrit a 2.0x. La 6/6 est la seule des trois a taper.
#
# Le terme descent existe deja et coute -0.0131 par episode, quand flat_support
# coute -0.47 et torque_limit_margin -1.11. Il est quarante fois trop faible
# pour peser : c'est le POIDS qui manque, pas le terme.
#
# C7 ne s'applique pas. La regle interdit d'augmenter un cout attache a
# l'ATTERRISSAGE, parce qu'on le satisfait toujours en n'atterrissant pas.
# descent_speed_cost se paie pendant toute la descente, par seconde : planer ne
# l'evite pas, un pied qui descend lentement descend longtemps.
#
# GARDE, et c'est le mode d'echec a surveiller : ne PAS lever le pied evite le
# cout. sole_height_p90 doit rester au-dessus de 0.030. swing_bonus (2.0) et
# l'horloge de demarche s'y opposent, mais la garde reste necessaire.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand"
export RHPS1_DEMAND_POWER=1.0 RHPS1_DEMAND_CAP=12.0 RHPS1_W_DEMAND=-0.03
# -40 et non -4 : dix fois, ce qui porte Episode_Reward/descent de -0.013 a
# ~-0.13. Modere volontairement -- cinq effondrements de type C7 sont dans ce
# journal, et meme un terme immunise merite d'etre monte par paliers.
export RHPS1_W_DESCENT=-40.0
# Limite inchangee a 0.12, qui est DEJA sous la mediane de la policy 0 (0.1476).
# Ce n'est pas la limite qui manquait.
export RHPS1_DESCENT_LIMIT=0.12
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-03_12-54-58 --agent.load-checkpoint model_5100.pt \
  --agent.max-iterations 2500
