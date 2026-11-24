#!/usr/bin/env bash
# vis_gate_full.sh <tag> <run> <first> [extra.env]: vis-lineage gates every 500 its
#   vismoke (CPU, 32 envs x 24 s): near and far without map failures, rob = far with the recipe's failures
#     -> logs/stairs_runs/visgate.txt
#   stop_eval (GPU, map failures off) -> stop.txt; oseq_eval every 1000 -> oseq.txt
#   prunes the checkpoints that are not multiples of 500 once gated (PRUNE=0 to keep them)
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=$1; RUN=$2; FIRST=$3; EX=${4:-$REPO/checkpoints/rhps1/vis3c_32500/vis3c_extra.env}
R=$REPO/logs/rsl_rl/rhps1_tracking/$RUN; OUT=$REPO/logs/stairs_runs; LOCK=$OUT/gate.lock
set -a; source "$EX"; set +a
ROB=$(grep -E "^RHPS1_VIS_(DROP|OUT_P|OUT_S|PERSON_P|PERSON_S|RAY_DROP|PHANTOM_P|PHANTOM_H)=" "$EX" | tr '\n' ' ')
NOFAIL="RHPS1_VIS_DROP=0 RHPS1_VIS_OUT_P=0 RHPS1_VIS_PERSON_P=0 RHPS1_VIS_RAY_DROP=0 RHPS1_VIS_PHANTOM_P=0"
for it in $(seq "$FIRST" 500 "${LAST:-99999}"); do
  until [ -f "$R/model_$it.pt" ]; do sleep 60; [ -d "$R" ] || exit 0; done; sleep 30
  (cd "$REPO"
   for m in near far rob; do
     if [ $m = near ]; then X="RHPS1_STAIRS_APPROACH_RAMP=1000000000 RHPS1_STAIRS_STAIR_PAUSE_P=0 RHPS1_STAIRS_PAUSE_P=0"
     else X="RHPS1_STAIRS_APPROACH_RAMP=0"; fi
     if [ $m = rob ]; then X="$X $ROB"; else X="$X $NOFAIL"; fi
     # shellcheck disable=SC2086
     O=$(SMOKE_N=32 SMOKE_T=24 DRIVE=vision bash scripts/stairs/probe.sh vismoke.py "$RUN" "model_$it.pt" -- RHPS1_STAIRS_FRAC=1.0 $X)
     echo "${TAG}_$it $m $(echo "$O" | grep '^SUMMARY')$(echo "$O" | grep -m1 -E 'Error|Traceback')" >> "$OUT/visgate.txt"
   done)
  exec 9>"$LOCK"; flock 9
  (cd "$REPO"; source scripts/stairs/common.sh; stairs_env ""
   # shellcheck disable=SC2086
   export $NOFAIL RHPS1_STAIRS_LVL_FILE=
   .venv/bin/python scripts/stairs/stop_eval.py "$RUN" "model_$it.pt" "${TAG}_$it" 2>&1 | grep -E -A5 "^STOP|Error|Traceback" >> "$OUT/stop.txt"
   if [ $((it % 1000)) -eq 0 ]; then
     .venv/bin/python scripts/stairs/oseq_eval.py "$RUN" "model_$it.pt" "${TAG}_$it" 2>&1 | grep -E -A9 "^OSEQ|Error|Traceback" >> "$OUT/oseq.txt"
   fi)
  flock -u 9
  [ "${PRUNE:-1}" = 1 ] && ls "$R"/model_*.pt | grep -vE "model_[0-9]*(000|500)\.pt" | xargs -r rm -f
  tail -n 3 "$OUT/visgate.txt"; df -h / | tail -1
done
