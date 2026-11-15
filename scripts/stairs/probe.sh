#!/usr/bin/env bash
# probe.sh <script.py> [args...] [-- VAR=VAL...]: run one of the mjlab probes (CPU) with the recipe environment.
#   probe.sh apsmoke.py <run> <ckpt>        deterministic rollout (SMOKE_N/SMOKE_T/SMOKE_DT), SUMMARY line
#   probe.sh stairs_terms.py <run> <ckpt> L per-term reward rates (STOCH=1 stochastic)
#   probe.sh kneestat.py <run> <ckpt> L     falls, loaded-knee angle, step length (BENCH_LEVELS, BENCH_PER)
# Extra VAR=VAL after "--" override the recipe (e.g. RHPS1_STAIRS_APPROACH_RAMP=0 for far spawns).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
P=$1; shift; ARGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARGS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
cd "$REPO"; stairs_env
export RHPS1_STAIRS_LVL_FILE= CLIP_ACTIONS=12 BENCH_NO_OVERRIDE=1 "$@"
nice -n 5 .venv/bin/python "$S/$P" "${ARGS[@]}" 2>&1 | grep -v "^Module\|Warn"
