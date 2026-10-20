#!/usr/bin/env bash
# First pass of a slot's arm_posture: stand an ONNX in mc_mujoco (bench slot 6, ex-4, command 0,
# 10 s) and print the arm_posture block of its standing fixed point.
#   scripts/tools/arm_posture_bench.sh <policy.onnx>
# Slot 4's ONNX is restored on exit. Refuses to start if mc_mujoco is running.
set -u
cd "$(dirname "$0")/../.."
O=$1
P=/home/lmoussafir/install/share/NewRLQPController/policy/rhps1_bwc_x10.onnx
B=$(mktemp -d)
cp "$P" "$B/bench.onnx"
trap 'cp "$B/bench.onnx" "$P"; echo "# slot 6 (banc) restaure ($(md5sum "$P" | cut -c1-8))"; rm -rf "$B"' EXIT
cp "$O" "$P"
touch "$B/mark"
RLQP_BENCH_DELAY_MS=30 RLQP_BENCH_CMD_TAU=0.3 bash scripts/tools/mc_mujoco_bench.sh 6 0 0 0 10 > /dev/null 2>&1
L=$(find ~/logs/mc_rtc -name 'mc-control-NewRLQPController-2*.bin' -newer "$B/mark" | sort | tail -1)
[ -z "$L" ] && { echo "aucun log"; exit 1; }
python3 scripts/tools/arm_posture.py "$L"
