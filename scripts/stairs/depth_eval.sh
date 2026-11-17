#!/usr/bin/env bash
# depth_eval.sh <student.pt | teacher> [label] [-- VAR=VAL...]: pinned stairs rollouts with the head camera (CPU).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
P=$1; shift; ARGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARGS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
cd "$REPO"; stairs_env "${STAIRS_ENV:-}"
export RHPS1_DEPTH=1 RHPS1_HEAD_HOLD=1 RHPS1_STAIRS_LVL_FILE= RHPS1_STAIRS_MODE_FIXED=0 CUDA_VISIBLE_DEVICES= "$@"
nice -n 5 .venv/bin/python "$S/depth_eval.py" "$P" "${ARGS[@]}" 2>&1 | grep -A20 '^DEPTHEVAL\|Traceback\|Error' | grep -v 'linesearch\|warn_overflow'
