#!/usr/bin/env bash
# export.sh <run> <ckpt> [out.onnx]: ONNX of a stairs checkpoint (CPU) + torch/ONNX parity on 8 random observations.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT=${3:-$CAND/$(basename $1)_$(basename $2 .pt).onnx}
cd "$REPO"; stairs_env
export RHPS1_STAIRS_LVL_FILE= CLIP_ACTIONS=12 CUDA_VISIBLE_DEVICES= \
  MJLAB_TASK=Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start MJLAB_LOG_ROOT=logs/rsl_rl/rhps1_tracking
.venv/bin/python scripts/tools/export_onnx.py "$1" "$2" "$OUT" 2>&1 | grep -vi warn | tail -2
.venv/bin/python "$S/parity.py" "$1" "$2" "$OUT" 2>&1 | grep -vi warn | grep PARITY
echo "$OUT"
