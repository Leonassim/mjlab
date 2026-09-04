#!/usr/bin/env bash
# Rejouer EXACTEMENT la politique installee en slot 11 de mc_mujoco.
#
# Run 2026-09-03_21-08-16 model_12099, l'onnx correspondant etant
# share/NewRLQPController/policy/rhps1_softland_it12099.onnx.
#
# Les variables doivent correspondre a l'entrainement, sinon la configuration
# construite n'est pas celle du checkpoint et le chargement echoue ou, pire,
# reussit sur un plant different.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand"
export RHPS1_DEMAND_POWER=1.0 RHPS1_DEMAND_CAP=12.0 RHPS1_W_DEMAND=-0.03
export RHPS1_W_DESCENT=-120.0
export RHPS1_W_SWINGBONUS=3.0
export RHPS1_SWINGBONUS_H=0.05
exec uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file logs/rsl_rl/rhps1_velocity/2026-09-03_21-08-16/model_12099.pt "$@"
