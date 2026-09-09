#!/usr/bin/env bash
# P3 : corriger la posture. UNE deviation depuis model_19799.
#
# Defaut signale par Leo et mesure : les jambes s'ecartent de 4 a 5 degres de
# roulis de hanche a l'arret, IDENTIQUEMENT en simulation et au deploiement.
# Cause : crotch_proximity (-0.478) pousse les cuisses a s'ecarter, et
# feet_distance (-0.001) ne le retient pas -- il ne se declenche jamais.
#
# Le seuil passe de 0.06 a 0.04. La policy 0, dont la posture convient, n'a pas
# ce terme du tout ; on ne le retire pas pour autant, l'evitement
# d'auto-collision reste utile.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand+bwcref"
export RHPS1_W_BWCREF=1e-9
export RHPS1_CLOCK_SLOW=0.9 RHPS1_CLOCK_FAST=0.9
export RHPS1_UPPER_SCALE=0.0007
export RHPS1_W_DESCENT=-120.0 RHPS1_DESCENT_LIMIT=0.12
export RHPS1_W_SWINGBONUS=3.0 RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_FLATSUP=-10.0 RHPS1_W_ANKLE_ROLL=-0.0008
export RHPS1_CROTCH_MINDIST=${RHPS1_CROTCH_MINDIST:-0.04}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-07_14-40-21 --agent.load-checkpoint model_19799.pt \
  --agent.max-iterations 4000
