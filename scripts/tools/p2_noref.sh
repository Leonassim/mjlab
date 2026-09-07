#!/usr/bin/env bash
# P2 : garder ce que P1 a decouvert, retirer ce qui l'empechait.
#
# P1 a montre deux choses sur 12000 iterations. La cadence BWC (horloge a 0.9 s)
# tient tout du long et fait le gain reel : 2.3 fois plus de pas, descente a la
# pose 0.171 -> 0.128, pic de charge 1.62 -> 1.43 le poids. Et la reference, elle,
# a ete ABANDONNEE par la politique -- refErr est remonte de 0.065 a 0.087 malgre
# un std deux fois plus exigeant -- et c'est precisement en l'abandonnant que le
# lever de pied est remonte de 0.009 a 0.037.
#
# A la fidelite atteignable (3.5 deg d'erreur RMS sur des amplitudes de 14-18),
# la reference ECRASE le lever : la hauteur du pied est une petite difference de
# grands mouvements articulaires. Elle a servi a DECOUVRIR la bonne cadence, pas
# a rester dans le bareme.
#
# Poids 1e-9 et non 0 : le RewardManager saute les termes a poids nul et ils ne
# loguent plus rien. On garde refErr comme diagnostic.
#
# Depart model_12750, seul checkpoint de P1b avec lever > 0.030 ET chutes < 0.05
# (lift 0.0350, falls 0.0278, impact 0.0914). Au-dela, le lever continue de
# monter mais les chutes explosent : 0.041 de lever pour 0.25 de chutes a 14400.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand+bwcref"
export RHPS1_W_BWCREF=1e-9
export RHPS1_CLOCK_SLOW=0.9 RHPS1_CLOCK_FAST=0.9
export RHPS1_UPPER_SCALE=0.0007
export RHPS1_DEMAND_POWER=1.0 RHPS1_DEMAND_CAP=12.0 RHPS1_W_DEMAND=-0.03
export RHPS1_W_DESCENT=-120.0 RHPS1_DESCENT_LIMIT=0.12
export RHPS1_W_SWINGBONUS=3.0 RHPS1_SWINGBONUS_H=0.05
# Pied a plat. Retirer la reference n'a PAS suffi : le talon est remonte a 0.80
# pendant le transitoire de reprise puis redescend, 0.556 a +225 iterations et
# toujours en baisse. -10 porte flat_support de -0.61 a environ -2.5, au-dessus
# des penalites de cheville (-0.85 et -1.07) qui l'ecrasaient.
export RHPS1_W_FLATSUP=${RHPS1_W_FLATSUP:--10.0}
# Les DEUX moities du meme rapport de forces. flat_support a -10 seul a produit
# une egalite (-3.592 contre -3.507 de cheville) : la politique paie exactement
# ce qu'elle gagne a aplatir, donc elle ne le fait pas. Relacher le roulis fait
# basculer.
export RHPS1_W_ANKLE_ROLL=${RHPS1_W_ANKLE_ROLL:--0.0008}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-07_12-07-19 --agent.load-checkpoint model_13050.pt \
  --agent.max-iterations 6000
