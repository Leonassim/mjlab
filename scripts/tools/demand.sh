#!/usr/bin/env bash
# T2 : payer la demande de couple avant ecretage. UNE deviation depuis la 6/6.
#
# Depart 2026-09-01_17-45-07 model_4500 -- verifie sur un plateau avant reprise
# (C4) : fell_down 0.0000, sole_height_p90 0.0329->0.0368 et mean_reward
# 78->84 monotones sur 4200-4800, aucun transitoire.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand"
export RHPS1_W_DEMAND=${RHPS1_W_DEMAND:--0.04}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
# video_interval est en pas de politique, pas en iterations : 24 pas par
# iteration a 4096 envs, donc 6000 = un clip toutes les 250 iterations.
# 600 pas a 5 ms = 3 s.
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-01_17-45-07 --agent.load-checkpoint model_4500.pt \
  --agent.max-iterations 2500
