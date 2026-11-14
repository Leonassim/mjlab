#!/usr/bin/env bash
# Rejouer la politique installee en slot 12 de mc_mujoco.
# Run 2026-09-07_14-40-21 model_13950, cadence BWC (horloge 0.9 s).
#
# Seules comptent ici les variables qui changent l'ESPACE D'OBSERVATION ou
# l'ECHELLE D'ACTION : l'horloge (gait_phase alimente 20 des 530 dimensions) et
# l'echelle du haut du corps. Les poids de recompense n'affectent pas le
# comportement en lecture.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand+bwcref"
export RHPS1_W_BWCREF=1e-9
export RHPS1_CLOCK_SLOW=0.9 RHPS1_CLOCK_FAST=0.9
export RHPS1_UPPER_SCALE=0.0007
export RHPS1_W_DESCENT=-120.0 RHPS1_W_SWINGBONUS=3.0 RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_FLATSUP=-10.0 RHPS1_W_ANKLE_ROLL=-0.0008
exec uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file logs/rsl_rl/rhps1_velocity/2026-09-07_14-40-21/model_13950.pt "$@"
