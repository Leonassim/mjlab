#!/usr/bin/env bash
# train_s50.sh <tag> [<load_run> <load_ckpt>]: the 50 Hz stairs recipe (s50e.env), 3584 envs, wandb + 3 s clips.
#   no load:  from scratch as s50 (2026-11-14_22-21-43): entropy 0.002, ramps over 72000 env steps.
#   load:     resume as s50e (2026-11-15_10-23-12): entropy 0, ramps frozen at their end.
# s50 used entropy 0.002 for 3000 its and its std rose 1.0 -> 1.75: resume with this script after ~1000-3000 its.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TAG=$1; LR=${2:-}; LC=${3:-}
cd "$REPO"
stairs_env "${STAIRS_ENV:-}"   # STAIRS_ENV: another recipe file (default s50e.env)
export RHPS1_STAIRS_LVL_FILE=$RUNS/${TAG}_lvl.pt
RESUME=(); ENT=0.0
if [ -z "$LR" ]; then
  export RHPS1_STAIRS_LAND_RAMP_OFF=0 RHPS1_STAIRS_APPROACH_OFF=0 RHPS1_STAIRS_BWC_RAMP=72000 \
    RHPS1_STAIRS_SAT_T0=1.0 RHPS1_STAIRS_SAT_RAMP=72000
  rm -f "$RHPS1_STAIRS_LVL_FILE"; ENT=0.002
else
  RESUME=(--agent.resume True --agent.load-run "$LR" --agent.load-checkpoint "$LC")
fi
[ -n "${DRY:-}" ] && { echo "DRY: $(env | grep -c "^RHPS1_") RHPS1 vars, LVL_MAX ${RHPS1_STAIRS_LVL_MAX:-}, env ${STAIRS_ENV:-s50e.env}, entropy $ENT, resume ${RESUME[*]:-none}, lvl $RHPS1_STAIRS_LVL_FILE, decimation $RHPS1_DECIMATION"; exit 0; }
nohup uv run train Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start --env.scene.num-envs "${NUM_ENVS:-3584}" \
  --env.commands.motion.motion-file "$REPO/$MJLAB_MOTION_FILE" \
  --video True --video-interval 6000 --video-length 150 "${RESUME[@]}" \
  --agent.max-iterations "${ITERS:-20000}" --agent.seed "${SEED:-11}" --agent.save-interval 250 \
  --agent.algorithm.gamma 0.99 --agent.algorithm.lam 0.95 --agent.algorithm.entropy-coef $ENT \
  --agent.clip-actions 12 > "$RUNS/$TAG.log" 2>&1 &
echo "$TAG PID $! log $RUNS/$TAG.log"
