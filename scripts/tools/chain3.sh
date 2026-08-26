#!/usr/bin/env bash
# Attend la fin du balayage de P5, remplace la consolidation par P6, puis rend
# la main a la consolidation.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
BASE=p0+rand+lift+steplen+freevel+freeroll+dense+calm+stable+soleclear+slowstep+softland+landtime+groundtax+freearms+impactladder+wide
OUT=logs/probes

while pgrep -f "sweep_eval.py 2026-08-26_17-49-40" > /dev/null; do sleep 20; done
sleep 30
pkill -f "chain2.sh" 2>/dev/null
pkill -f "bin/train Mjlab-Velocity-Flat-RHPS1" 2>/dev/null
sleep 20

echo "$(date +%H:%M:%S) -> P6_flatpay_dt (scale 0.12, poids 0.4, /step_dt)" >> "$OUT/chain.log"
scripts/tools/probe.sh P6_flatpay_dt "$BASE+flatpay" 450 >> "$OUT/chain.log" 2>&1 \
  || echo "$(date +%H:%M:%S) !! P6 a echoue" >> "$OUT/chain.log"

echo "$(date +%H:%M:%S) -> consolidation" >> "$OUT/chain.log"
export RHPS1_ABLATION="$BASE"
export RHPS1_CLEAR_TARGET=0.027 RHPS1_FREEVEL_SHARE=0.6 RHPS1_FREEVEL_STD=0.45
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.resume True --agent.load-run 2026-08-26_00-21-43 \
  --agent.load-checkpoint model_18000.pt \
  --agent.max-iterations 6000 >> "$OUT/consolidation.train.log" 2>&1
