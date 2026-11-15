# common.sh: paths and recipe environment shared by the stairs scripts (sourced, bash).
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$S/../.." && pwd)
RUNS=${RHPS1_RUNS:-$REPO/logs/stairs_runs}  # launcher logs, level files, bench results
CAND=${RHPS1_CAND:-$REPO/logs/stairs_cand}  # exported ONNX candidates
RLC=${RL_CONTROLLER_DIR:-$HOME/src/rl_controller}
mkdir -p "$RUNS" "$CAND"
# stairs_env [file]: export a recipe's settings (default s50e.env, the 2026-11-15 50 Hz run).
stairs_env() { set -a; source "${1:-$S/s50e.env}"; set +a; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}; }
