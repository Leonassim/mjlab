#!/usr/bin/env bash
# ohbench.sh <onnx> <label>: relbench.sh's 12 mc_mujoco runs (30 s each) + overhang.py, where the loaded sole sits on
# the steps (descent and climb). Same summary line as relbench, then an OVERHANG line. Logs deleted unless KEEP_LOGS=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export RLQP_BENCH_POLICY_PATH=$1 RLQP_BENCH_POLICY_STEP=${RLQP_BENCH_POLICY_STEP:-0.02}; LOGS=(); NPZ=()
OH=$RUNS/oh; mkdir -p "$OH"
for H in ${HS:-0.04 0.05 0.06}; do for V in ${VSTS:-2.0 3.0 4.0 5.0}; do
  L=$(VST=$V bash "$S/headless.sh" $H 0.15 ${SECS:-30} 2>/dev/null | tail -1); LOGS+=("$L")
  Z=$OH/${2}_h${H}_v${V}.npz
  [ -f "$L" ] && (cd /tmp && python3 "$S/overhang.py" export "$L" "$Z") && NPZ+=("$Z")
done; done
(cd /tmp && timeout 1800 python3 "$S/relsum.py" "$2" "${LOGS[@]}")
(cd "$REPO" && .venv/bin/python "$S/overhang.py" stats "$2" "${NPZ[@]}")
[ -n "${KEEP_LOGS:-}" ] || rm -f "${LOGS[@]}"
