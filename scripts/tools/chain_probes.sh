#!/usr/bin/env bash
# Enchaine les sondes restantes puis tombe dans une run longue. Le GPU ne doit
# jamais rester libre : dix fois de suite une sonde s'est terminee sans que rien
# ne reprenne derriere, et chaque trou coute une heure de calcul.
#
# Une sonde qui echoue (rung absent, config invalide) ne casse pas la chaine :
# on la note et on passe a la suivante.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
BASE=p0+rand+lift+steplen+freevel+freeroll+dense+calm+stable+soleclear+slowstep+softland+landtime+groundtax+freearms+impactladder+wide
OUT=logs/probes; mkdir -p "$OUT"
echo "chaine demarree $(date +%H:%M:%S)" >> "$OUT/chain.log"

run_probe() {
  local name=$1 abl=$2 iters=$3
  echo "$(date +%H:%M:%S) -> $name ($abl)" >> "$OUT/chain.log"
  scripts/tools/probe.sh "$name" "$abl" "$iters" >> "$OUT/chain.log" 2>&1 \
    || echo "$(date +%H:%M:%S) !! $name a echoue" >> "$OUT/chain.log"
}

run_probe P3_periodlive "$BASE+periodlive" 450
run_probe P4_flatpay    "$BASE+flatpay"    450

# Filet : tant que rien d'autre ne tourne, consolider. Une run longue vaut mieux
# qu'un GPU libre, et elle s'interrompt proprement quand je lance autre chose.
echo "$(date +%H:%M:%S) -> run longue de consolidation" >> "$OUT/chain.log"
export RHPS1_ABLATION="$BASE"
export RHPS1_CLEAR_TARGET=0.027 RHPS1_FREEVEL_SHARE=0.6 RHPS1_FREEVEL_STD=0.45
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.resume True --agent.load-run 2026-08-26_00-21-43 \
  --agent.load-checkpoint model_18000.pt \
  --agent.max-iterations 6000 >> "$OUT/consolidation.train.log" 2>&1
