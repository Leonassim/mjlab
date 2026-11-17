#!/usr/bin/env bash
# vis_gate_loop.sh <run> <tag> <first_it>: every 500 its, deterministic mjlab rollouts of the vision student on the
# onboard map (vismoke.py, DRIVE=vision), near and far spawns, 32 envs x 24 s, into $RUNS/visgate.txt.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DIR=$REPO/logs/rsl_rl/rhps1_tracking/$1
for it in $(seq $3 500 999999); do
  until [ -f $DIR/model_$it.pt ]; do sleep 60; [ -d $DIR ] || exit 0; done
  sleep 30
  for m in near far; do
    if [ $m = near ]; then X="RHPS1_STAIRS_APPROACH_RAMP=1000000000 RHPS1_STAIRS_STAIR_PAUSE_P=0 RHPS1_STAIRS_PAUSE_P=0"; else X="RHPS1_STAIRS_APPROACH_RAMP=0"; fi
    OUT=$(SMOKE_N=32 SMOKE_T=24 DRIVE=vision bash "$S/probe.sh" vismoke.py $1 model_$it.pt -- RHPS1_STAIRS_FRAC=1.0 $X)
    echo "$2_$it $m $(echo "$OUT" | grep '^SUMMARY') | $(echo "$OUT" | grep '^map')" >> "$RUNS/visgate.txt"
  done
  tail -2 "$RUNS/visgate.txt"
done
