#!/usr/bin/env bash
# gate_loop.sh <run> <tag> <first_it>: every 500 its, export + mc_mujoco relbench (gate.txt) and deterministic mjlab rollouts
# near (usual spawn) and far (full approach) (detcheck.txt), in $RUNS.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DIR=$REPO/logs/rsl_rl/rhps1_tracking/$1
for it in $(seq $3 500 999999); do
  until [ -f $DIR/model_$it.pt ]; do sleep 60; [ -d $DIR ] || exit 0; done
  sleep 30
  O=$(bash "$S/export.sh" $1 model_$it.pt "$CAND/$2_$it.onnx" | tail -1)
  bash "$S/relbench.sh" "$O" $2_$it 2>&1 | tail -13 >> "$RUNS/gate.txt"
  for m in near far; do
    if [ $m = near ]; then X="RHPS1_STAIRS_APPROACH_RAMP=1000000000 RHPS1_STAIRS_STAIR_PAUSE_P=0 RHPS1_STAIRS_PAUSE_P=0"; else X="RHPS1_STAIRS_APPROACH_RAMP=0"; fi
    echo "$2_$it $m $(SMOKE_N=32 SMOKE_T=24 SMOKE_DT=100 bash "$S/probe.sh" apsmoke.py $1 model_$it.pt -- RHPS1_STAIRS_FRAC=1.0 $X | grep '^SUMMARY')" >> "$RUNS/detcheck.txt"
  done
  grep "RELBENCH $2_$it" "$RUNS/gate.txt"; tail -2 "$RUNS/detcheck.txt"
done
