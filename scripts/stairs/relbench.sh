#!/usr/bin/env bash
# relbench.sh <onnx> <label>: 12 mc_mujoco runs (HS 4/5/6 cm x VSTS start 2/3/4/5 s), forward to the end, one summary line.
# 50 Hz policies need RLQP_BENCH_POLICY_STEP=0.02 (default here); 200 Hz ones RLQP_BENCH_POLICY_STEP=0.005.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export RLQP_BENCH_POLICY_PATH=$1 RLQP_BENCH_POLICY_STEP=${RLQP_BENCH_POLICY_STEP:-0.02}; LOGS=()
for H in ${HS:-0.04 0.05 0.06}; do for V in ${VSTS:-2.0 3.0 4.0 5.0}; do
  L=$(VST=$V bash "$S/headless.sh" $H 0.15 25 2>/dev/null | tail -1); LOGS+=("$L")
done; done
cd /tmp && timeout 1800 python3 "$S/relsum.py" "$2" "${LOGS[@]}"
