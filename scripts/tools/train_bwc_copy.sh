#!/usr/bin/env bash
# Resume the BWC-copy tracking training with slot 10's configuration.
#   scripts/tools/train_bwc_copy.sh <load_run> <load_ckpt> <iterations> <log> [EXTRA=val ...]
# Slot 10 = run 2026-10-14_10-55-12 model_31000. Extra env assignments add one
# deviation on top (e.g. RHPS1_FREE_ARMS=1). Prints the trainer PID.
set -eu
cd "$(dirname "$0")/../.."
RUN=$1; CK=$2; ITERS=$3; LOG=$4; shift 4
export CUDA_VISIBLE_DEVICES=0 \
  RHPS1_POSTURE_K=1600 RHPS1_POSTURE_K_HI=0 RHPS1_FOOT_FRICTION=1.0 RHPS1_ACTION_SCALE_MULT=10 \
  RHPS1_ACTION_DELAY_MS=45 RHPS1_CMD_TAU=0.3 \
  RHPS1_DOUBLE_SUPPORT_W=2 RHPS1_HEADING_W=1 RHPS1_NOSLIP_W=1 RHPS1_MIRROR=1 \
  RHPS1_STAND_SEGMENTS=1 RHPS1_CLEAN_LABELS=1 RHPS1_STANDING_START=0.8 \
  RHPS1_ROBUST=obs,bias,gains,friction,com \
  RHPS1_FEET_HEIGHT_W=5 RHPS1_FEET_HEIGHT_STD=0.015 \
  MJLAB_MOTION_FILE=docs/bwc_motion_v3.npz
for kv in "$@"; do export "$kv"; done
# MJLAB_MOTION_FILE=docs/... as an extra swaps the reference (e.g. the BWC with
# the human upper-body swing, blend_reference.py). TRAIN_ARGS adds CLI flags
# (e.g. "--agent.seed 7 --agent.save-interval 250").
nohup uv run train Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --env.commands.motion.motion-file "$PWD/$MJLAB_MOTION_FILE" \
  --video True --video-interval 6000 --video-length 600 \
  --agent.resume True --agent.load-run "$RUN" --agent.load-checkpoint "$CK" \
  --agent.max-iterations "$ITERS" ${TRAIN_ARGS:-} > "$LOG" 2>&1 &
echo $!
