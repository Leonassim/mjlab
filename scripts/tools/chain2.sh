#!/usr/bin/env bash
# P5 puis consolidation. Meme regle que chain_probes.sh : rien ne s'arrete sans
# que la suite soit deja lancee.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
BASE=p0+rand+lift+steplen+freevel+freeroll+dense+calm+stable+soleclear+slowstep+softland+landtime+groundtax+freearms+impactladder+wide
OUT=logs/probes

# scale 0.12 rad, la valeur MESUREE de l'inclinaison a la pose en marche
# (tiltTD 0.105-0.142 au balayage de P4). A 0.06 le bonus versait exp(-4) =
# 1.8% de son maximum : hors de sa zone de reponse, donc sans effet.
export RHPS1_FLATPAY_SCALE=0.12
echo "$(date +%H:%M:%S) -> P5_flatpay12 (scale 0.12)" >> "$OUT/chain.log"
scripts/tools/probe.sh P5_flatpay12 "$BASE+flatpay" 450 >> "$OUT/chain.log" 2>&1 \
  || echo "$(date +%H:%M:%S) !! P5 a echoue" >> "$OUT/chain.log"

echo "$(date +%H:%M:%S) -> consolidation" >> "$OUT/chain.log"
unset RHPS1_FLATPAY_SCALE
export RHPS1_ABLATION="$BASE"
export RHPS1_CLEAR_TARGET=0.027 RHPS1_FREEVEL_SHARE=0.6 RHPS1_FREEVEL_STD=0.45
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.resume True --agent.load-run 2026-08-26_00-21-43 \
  --agent.load-checkpoint model_18000.pt \
  --agent.max-iterations 6000 >> "$OUT/consolidation.train.log" 2>&1
