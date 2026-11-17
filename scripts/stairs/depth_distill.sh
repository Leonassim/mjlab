#!/usr/bin/env bash
# depth_distill.sh <tag> [depth_distill.py args]: DAgger of the depth-image student from s50e 15000 (default teacher).
#   ENVS (default 2048): number of envs; the teacher's level file is cut to it (levels only load on a size match).
#   STAIRS_ENV: recipe file (default s50e.env); TEACHER: <run>/<ckpt> under logs/rsl_rl/rhps1_tracking.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TAG=$1; shift
cd "$REPO"
stairs_env "${STAIRS_ENV:-}"
export RHPS1_DEPTH=1 RHPS1_HEAD_HOLD=1 RHPS1_STAIRS_LVL_FILE=$RUNS/${TAG}_lvl.pt
ENVS=${ENVS:-2048}
[ -f "$RHPS1_STAIRS_LVL_FILE" ] || .venv/bin/python -c "import torch,sys; d=torch.load(sys.argv[1]); n=int(sys.argv[3]); torch.save({k: v[:n].clone() for k, v in d.items()}, sys.argv[2])" \
  checkpoints/rhps1/s50e_50hz_15000/s50e_lvl.pt "$RHPS1_STAIRS_LVL_FILE" "$ENVS"
T=logs/rsl_rl/rhps1_tracking/${TEACHER:-2026-11-15_10-23-12/model_15000.pt}
[ -n "${DRY:-}" ] && { echo "DRY: teacher $T, lvl $RHPS1_STAIRS_LVL_FILE, $(env | grep -c '^RHPS1_') RHPS1 vars, args $*"; exit 0; }
nohup .venv/bin/python "$S/depth_distill.py" --teacher "$T" --tag "$TAG" --envs "$ENVS" "$@" > "$RUNS/$TAG.log" 2>&1 &
echo "$TAG PID $! log $RUNS/$TAG.log"
