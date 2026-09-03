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
# -120, deuxieme palier. A -40 la mesure en marche donne p99 0.346 -> 0.287,
# soit -16% seulement, et la force de pic EMPIRE (2.57x -> 2.72x le poids) :
# baisser l'elan sans allonger le contact ne change pas Delta_p/Delta_t.
#
# Hypothese "jambe raide" TESTEE ET FAUSSE : la policy 0 atterrit avec MOINS de
# flexion de genou que nous (2.0 deg contre 2.9) et tape moins fort. La flexion
# n'est pas le levier, donc pas de terme de compliance.
#
# Le vrai mecanisme est la cadence : 411 poses contre 1737 pour la policy 0 sur
# la meme duree, soit quatre fois plus de vol par pas et d'autant plus de
# hauteur a retomber. C'est le prix direct de la cadence BWC obtenue en reglant
# D5, et le seul moyen de garder les deux est de freiner activement la descente.
# Le levier est donc le bon, il est trop faible : Episode_Reward/descent vaut
# -0.10 quand flat_support vaut -0.47.
export RHPS1_W_DESCENT=-120.0
# Limite inchangee a 0.12, qui est DEJA sous la mediane de la policy 0 (0.1476).
# Ce n'est pas la limite qui manquait.
export RHPS1_DESCENT_LIMIT=0.12
# Contrepoids. Le balayage a -120 donne 5 criteres sur 6 : impact 0.1441 ->
# 0.1230, mais lever de pied 0.0326 -> 0.0271, sous le seuil de 0.030 -- et
# UNIQUEMENT a vx=0.10, la marche la plus lente ; a 0.20 et 0.30 il vaut 0.036.
# On a paye l'impact avec le lever, ce que le cout de descente encourage.
#
# swing_height_bonus_dense est l'oppose exact : il paie la HAUTEUR pendant le
# vol, en continu, donc il est C7-immun comme lui et s'y oppose directement.
# 3.0 au lieu de 2.0. Les deux peuvent coexister -- monter plus haut ET
# descendre plus lentement est possible tant qu'il reste du temps de vol, et la
# periode est a 0.73 s contre 0.90 pour le BWC.
# 2.5, a mi-chemin. Trois runs encadrent le compromis :
#
#                       vol   descente    lever   couples
#   6/6                 2.0        -4    0.0326    0.0178
#   T5b                 2.0      -120    0.0271    0.0212
#   T6b                 3.0      -120    0.0386    0.0418   seuils 0.030 / 0.030
#
# Le cout de descente seul coute peu (+19% de couple a lever constant). C'est le
# bonus de vol porte a 3.0 qui a double la saturation : lever plus haut coute du
# couple, mecaniquement. A 2.5 l'interpolation donne lever ~0.033 et couples
# ~0.031, les deux au seuil -- la tension est reelle et se joue entre 2.3 et 2.5.
export RHPS1_W_SWINGBONUS=2.5
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-03_21-08-16 --agent.load-checkpoint model_12099.pt \
  --agent.max-iterations 3000
