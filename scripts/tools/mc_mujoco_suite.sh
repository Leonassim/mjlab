#!/usr/bin/env bash
# Walk an ONNX in mc_mujoco in the six directions, at several push times.
#   scripts/tools/mc_mujoco_suite.sh <policy.onnx> [push times, default "3.0 4.5"]
# The ONNX is swapped into bench slot 6 (ex-4, rhps1_bwc_x10.onnx) for the run and the
# original restored on exit. 30 ms target delay, command filter 0.3 s. Refuses to
# start, or stops, if someone else's mc_mujoco is running.
set -u
cd "$(dirname "$0")/../.."
O=$1; PUSHES=${2:-"3.0 4.5"}
P=/home/lmoussafir/install/share/NewRLQPController/policy/rhps1_bwc_x10.onnx
B=$(mktemp -d)
cp "$P" "$B/bench.onnx"
trap 'cp "$B/bench.onnx" "$P"; echo "slot 6 (banc) restaure ($(md5sum "$P" | cut -c1-8))"; rm -rf "$B"' EXIT
cp "$O" "$P"
for c in "0.3 0 0" "-0.3 0 0" "0 0.3 0" "0 -0.3 0" "0 0 0.3" "0 0 -0.3"; do
  read -r vx vy wz <<<"$c"
  for st in $PUSHES; do
    pgrep -x mc_mujoco >/dev/null && { echo "mc_mujoco occupe, arret"; exit 1; }
    touch "$B/mark"
    RLQP_BENCH_DELAY_MS=30 RLQP_BENCH_CMD_TAU=0.3 RLQP_BENCH_VEL_START_S=$st \
      bash scripts/tools/mc_mujoco_bench.sh 6 "$vx" "$vy" "$wz" 15 > /dev/null 2>&1
    L=$(find ~/logs/mc_rtc -name 'mc-control-NewRLQPController-2*.bin' -newer "$B/mark" | sort | tail -1)
    echo -n "[$vx $vy $wz, poussee +$st s] "
    python3 scripts/tools/mc_mujoco_dirs.py "$L"
  done
done
