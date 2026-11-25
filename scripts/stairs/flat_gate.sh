#!/usr/bin/env bash
# flat_gate.sh <tag> <run> [first]: flat-walker gates every 500 its (slot 18 as the yardstick)
#   GATE_ENV=scripts/stairs/<recipe>.env: the run's recipe (default s50e.env: wrong for decimation or action-mode changes)
#   oseq_eval (per axis: speed ratio, leak, roll, min ankle width, stop slip) -> logs/stairs_runs/oseq.txt
#   stop_eval (roll, hike, swing-knee excursion, foot lift, cadence, slip, stagger) -> logs/stairs_runs/stop.txt
#   prunes the checkpoints that are not multiples of 500 once gated (PRUNE=0 to keep them)
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=$1; RUN=$2; FIRST=${3:-500}
R=$REPO/logs/rsl_rl/rhps1_tracking/$RUN; OUT=$REPO/logs/stairs_runs
for it in $(seq "$FIRST" 500 "${LAST:-99999}"); do
  until [ -f "$R/model_$it.pt" ]; do sleep 60; [ -d "$R" ] || exit 0; done; sleep 20
  exec 9>"$OUT/gate.lock"; flock 9
  (cd "$REPO"; source scripts/stairs/common.sh; stairs_env "${GATE_ENV:-}"; export RHPS1_STAIRS_LVL_FILE=
   .venv/bin/python scripts/stairs/oseq_eval.py "$RUN" "model_$it.pt" "${TAG}_$it" 2>&1 | grep -E -A9 "^OSEQ|Error|Traceback" >> "$OUT/oseq.txt"
   .venv/bin/python scripts/stairs/stop_eval.py "$RUN" "model_$it.pt" "${TAG}_$it" 2>&1 | grep -E -A5 "^STOP|Error|Traceback" >> "$OUT/stop.txt")
  flock -u 9
  [ "${PRUNE:-1}" = 1 ] && ls "$R"/model_*.pt | grep -vE "model_[0-9]*(000|500)\.pt" | xargs -r rm -f
  df -h / | tail -1
done
