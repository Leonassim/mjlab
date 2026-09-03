#!/usr/bin/env bash
# CONTROLE de T2 : exactement la meme reprise, SANS le terme demand.
#
# Regle "blamer le checkpoint d'abord" : un checkpoint fragile et une mauvaise
# recompense donnent la meme courbe. La reprise avec demand est partie a -6.76
# la ou la run d'origine etait a +80 a la meme iteration, avec une penalite de
# -0.19 -- donc soit c'est C4 (reprendre casse), soit c'est la variance du
# terme. Cette run repond, rien d'autre ne le peut.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-01_17-45-07 --agent.load-checkpoint model_4500.pt \
  --agent.max-iterations 150
