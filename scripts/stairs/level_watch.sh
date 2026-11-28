#!/usr/bin/env bash
# level_watch.sh <tag> <run> <recipe.env> <first> <stop_it>: every 500 its, append the stairs curriculum
# level / arrival_climb of <tag> to logs/stairs_runs/levels.txt; every 1000 its, export the ONNX
# (with the recipe env, else export fails silently) and push it to checkpoints/rhps1/<tag>_<it> for
# mc_mujoco benches. Exits after <stop_it>.
set -u
TAG=$1; RUN=$2; ENVF=$3; IT=$4; STOP=$5
REPO=$HOME/mjlab-rhps1; cd "$REPO"
R=logs/rsl_rl/rhps1_tracking/$RUN; LOG=logs/stairs_runs/$TAG.log; OUT=logs/stairs_runs/levels.txt
while [ "$IT" -le "$STOP" ]; do
  until [ -s "$R/model_$IT.pt" ]; do grep -q Traceback "$LOG" && exit 1; sleep 60; done
  sleep 20
  m() { tac "$LOG" | grep -m1 "$1" | grep -oE "[0-9.]+$"; }
  echo "LVL ${TAG}_$IT: level $(m 'stairs_levels/level:') step $(m 'step_height:') climb $(m 'arrival_climb:') descend $(m 'arrival_descend:')" >> "$OUT"
  if [ $((IT % 1000)) = 0 ]; then
    D=checkpoints/rhps1/${TAG}_$IT
    (mkdir -p "$D"; cp "$R/model_$IT.pt" "scripts/stairs/$ENVF" "$D/"
     source scripts/stairs/common.sh; stairs_env "scripts/stairs/$ENVF" >/dev/null 2>&1; export RHPS1_STAIRS_LVL_FILE=
     MJLAB_TASK=Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start MJLAB_LOG_ROOT=logs/rsl_rl/rhps1_tracking CUDA_VISIBLE_DEVICES="" \
       .venv/bin/python scripts/tools/export_onnx.py "$RUN" "model_$IT.pt" "$D/policy.onnx" 2>&1 | grep -E "wrote|Error"
     echo "$TAG $IT (run $RUN), recipe $ENVF. $(tail -1 "$OUT" | cut -d: -f2-)" > "$D/README.md"
     if [ -s "$D/policy.onnx" ]; then
       git pull -q --rebase origin main; git add "$D" && git commit -qm "Checkpoints: $TAG $IT with ONNX (stairs bench)" && git push -q origin main
       echo "LVL ${TAG}_$IT: ONNX pushed $(git log --oneline -1)" >> "$OUT"
     else echo "LVL ${TAG}_$IT: ONNX export FAILED" >> "$OUT"; fi)
  fi
  IT=$((IT + 500))
done
