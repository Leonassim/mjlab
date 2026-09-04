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
export RHPS1_DEMAND_POWER=1.0 RHPS1_DEMAND_CAP=12.0 RHPS1_W_DEMAND=-0.10
# -0.10 et non -0.03. Trois tentatives ont echoue sur la saturation de couple
# des jambes, et TOUTES portaient sur l'equilibre vol/descente :
#
#   T7   poids de vol 3.0 -> 2.5        demarche cassee (piaffement)
#   T8   cible de vol 0.05 -> 0.04      demarche cassee, meme signature
#   T9   descente -120 -> -80           couples PIRES (0.287 contre 0.17)
#
# Or le terme qui vise DIRECTEMENT la demande de couple etait reste a -0.03 tout
# du long. La reference valide le seuil : policy 0 a 0.0103, index 11 a 0.0418,
# donc quatre fois pire -- c'est un vrai defaut, pas un seuil arbitraire.
#
# Ce levier ne touche pas l'equilibre vol/descente, dont on sait maintenant
# qu'il est bistable et fragile des deux cotes.
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
# -80 et non -120. L'incitation a lever ne peut pas etre touchee (T7 et T8), donc
# c'est l'autre cote du compromis qui recule. La marge est de ce cote-la :
#
#   impact          0.1376   seuil 0.160    14% de marge, on peut en rendre
#   lever de pied   0.0386   seuil 0.030    marge, mais toucher au vol casse
#   couples jambes  0.0418   seuil 0.030    39% AU-DESSUS, rien a rendre
#
# Relacher la descente laisse le pied tomber un peu plus vite, ce qui coute de
# l'impact -- dont on a 14% de marge -- et rend du couple de jambe, dont on n'a
# aucune.
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
# 3.0, et NON 2.5. L'essai a 2.5 (T7) a casse la demarche : lever 0.0053,
# periode 1.58 s, reward -115 la ou T6b valait +85 au meme point. Le robot a
# arrete de lever le pied -- le mode d'echec annonce en lancant le cout de
# descente, declenche en baissant le contrepoids et non en montant le cout.
#
# Le systeme est BISTABLE, pas lineaire. J'avais interpole entre les deux runs
# connus pour predire "lever ~0.033, couples ~0.031" a 2.5 ; il n'y a pas de
# point intermediaire, le bonus de vol est soit au-dessus du seuil ou lever vaut
# la peine, soit en dessous et la politique bascule sur le piaffement.
#
#                       vol   descente    lever   couples
#   6/6                 2.0         -4   0.0326   0.0178
#   T5b                 2.0       -120   0.0271   0.0212
#   T6b                 3.0       -120   0.0386   0.0418
#   T7                  2.5       -120   0.0053   ---      casse
export RHPS1_W_SWINGBONUS=3.0
# Le levier contre la saturation de couple est donc la CIBLE, pas le poids.
# A 0.05 pour un vol realise de 0.048, le bonus tire encore et paie du couple
# pour trois millimetres. A 0.04 il sature : plus de gradient au-dessus, donc
# plus de poussee, sans retirer l'incitation a lever -- ce qui est precisement
# ce que retirer du poids a detruit. Regle deja ecrite : "une cible plafonnee a
# sa valeur n'a plus de gradient".
# Cible REMISE a 0.05. L'essai a 0.04 (T8) a casse la demarche exactement comme
# T7 : lever 0.0098, periode 1.65 s, reward -144 a 890 iterations, contre +85
# pour T6b au meme point. Les DEUX facons de reduire l'incitation a lever --
# le poids (T7) et la cible (T8) -- donnent la meme rupture. Ce n'est donc pas
# le parametre, c'est le seuil au-dela duquel lever ne vaut plus son cout de
# descente.
export RHPS1_SWINGBONUS_H=0.05
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-03_21-08-16 --agent.load-checkpoint model_12099.pt \
  --agent.max-iterations 3000
