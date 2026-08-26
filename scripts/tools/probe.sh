#!/usr/bin/env bash
# Une sonde : N iterations depuis UN checkpoint fixe, puis un balayage.
#
# Le point est la comparabilite. Les quinze runs du 24-26 aout partaient de
# quinze checkpoints differents avec des configurations qui se recouvraient, et
# aucun resultat n'etait comparable a un autre. Ici tout part du meme point et
# ne differe que par le rung teste.
#
#   probe.sh <nom> <chaine_ablation> [iterations]
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
NAME=$1; ABL=$2; ITERS=${3:-400}
BASE_RUN=${BASE_RUN:-2026-08-26_00-21-43}
BASE_CKPT=${BASE_CKPT:-model_18000.pt}
OUT=logs/probes; mkdir -p "$OUT"

while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 60; done
export RHPS1_ABLATION=$ABL
export RHPS1_CLEAR_TARGET=${RHPS1_CLEAR_TARGET:-0.027}
export RHPS1_FREEVEL_SHARE=0.6 RHPS1_FREEVEL_STD=0.45
# Pas de WANDB_MODE=offline : Leo suit les entrainements dans wandb, et le
# couper pour economiser un peu de surcharge sur des runs courts revient a
# retirer sa seule fenetre sur ce qui tourne.
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300

PREV_DIR=$(ls -dt logs/rsl_rl/rhps1_velocity/*/ 2>/dev/null | head -1)
START=$(date +%s)
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 7200 --video-length 600 \
  --agent.resume True --agent.load-run "$BASE_RUN" --agent.load-checkpoint "$BASE_CKPT" \
  > "$OUT/$NAME.train.log" 2>&1 &
TPID=$!
# Arret sur le checkpoint, pas sur --agent.max-iterations : ce drapeau compte des
# iterations SUPPLEMENTAIRES, pas une cible absolue. 18000+450 lui a fait viser
# 18450 iterations de plus, un jour et demi au lieu de quarante minutes -- et des
# sondes de durees differentes ne sont comparables a rien, ce qui est le seul
# interet de la methode.
BASE_IT=$(echo "$BASE_CKPT" | tr -dc 0-9)
TARGET=$(( BASE_IT + ITERS ))
# Attendre le repertoire que CE trainer cree, puis le checkpoint DEDANS. Un glob
# sur tous les repertoires trouve model_18450.pt d'une sonde precedente et
# declenche l'arret aussitot : P2 est morte en vingt secondes sans une iteration,
# et le balayage qui a suivi a mesure le run precedent en croyant mesurer P2.
NEW=""
for _ in $(seq 1 60); do
  C=$(ls -dt logs/rsl_rl/rhps1_velocity/*/ 2>/dev/null | head -1)
  [ -n "$C" ] && [ "$C" != "$PREV_DIR" ] && { NEW=$C; break; }
  sleep 20
done
[ -z "$NEW" ] && { echo "$NAME: aucun repertoire de run cree"; kill $TPID 2>/dev/null; exit 1; }
echo "$NAME: run $NEW, cible model_$TARGET.pt"
until [ -f "$NEW/model_$TARGET.pt" ] || ! kill -0 $TPID 2>/dev/null; do sleep 20; done
kill $TPID 2>/dev/null; sleep 15

RUN=$(basename "$NEW")
CKPT=$(ls "logs/rsl_rl/rhps1_velocity/$RUN"/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)
echo "sonde $NAME : run $RUN, checkpoint $CKPT, $(( ($(date +%s)-START)/60 )) min" | tee "$OUT/$NAME.txt"
.venv/bin/python scripts/tools/sweep_eval.py "$RUN" "model_$CKPT.pt" \
  --steps 900 --envs 256 --rand >> "$OUT/$NAME.txt" 2>&1
tail -8 "$OUT/$NAME.txt"
