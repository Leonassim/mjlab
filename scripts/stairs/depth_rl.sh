#!/usr/bin/env bash
# depth_rl.sh <tag> <dagger_student.pt> [depth_rl.py args]: PPO fine-tuning of the depth student (ENVS, default 2048).
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TAG=$1; ST=$2; shift 2
cd "$REPO"
stairs_env "${STAIRS_ENV:-}"
export RHPS1_DEPTH=1 RHPS1_HEAD_HOLD=1 RHPS1_DEPTH_RANDLEVEL=1 RHPS1_MIRROR=0 RHPS1_BOUND_W=0 \
  RHPS1_STAIRS_LVL_FILE=$RUNS/${TAG}_lvl.pt
ENVS=${ENVS:-2048}
[ -f "$RHPS1_STAIRS_LVL_FILE" ] || .venv/bin/python -c "import torch,sys; d=torch.load(sys.argv[1]); n=int(sys.argv[3]); torch.save({k: v[:n].clone() for k, v in d.items()}, sys.argv[2])" \
  checkpoints/rhps1/s50e_50hz_15000/s50e_lvl.pt "$RHPS1_STAIRS_LVL_FILE" "$ENVS"
nohup .venv/bin/python "$S/depth_rl.py" --student "$ST" --tag "$TAG" --envs "$ENVS" "$@" > "$RUNS/$TAG.log" 2>&1 &
echo "$TAG PID $! log $RUNS/$TAG.log"
