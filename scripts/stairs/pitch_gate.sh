#!/usr/bin/env bash
# pitch_gate.sh <tag> <run> <first> <lineage> <recipe.env>: backward-lean screen every 500 its (tardis 2026-11-27).
#   oseq_eval back and back+left only, chest CoM shifted -0.037 (RHPS1_COM_SHIFT_X, CoM randomisation off), as the
#   mc_mujoco candidate bench. Calibrated on gimpCT 10000/11000, gytdC 11000/11500: back+left pitch min ranks them as
#   mc_mujoco's back+left diagonal, ~1-2 deg milder. PASS = back+left pitch min >= PITCH_MIN (-3 deg).
#   A PASS that beats the lineage's best (logs/stairs_runs/pitch_best_<lineage>.txt) exports and pushes the ONNX to
#   checkpoints/rhps1/<tag>_<it>. Results -> logs/stairs_runs/pitch.txt.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TAG=$1; RUN=$2; FIRST=$3; LIN=$4; ENVF=$5
R=$REPO/logs/rsl_rl/rhps1_tracking/$RUN; OUT=$REPO/logs/stairs_runs; BEST=$OUT/pitch_best_$LIN.txt
for it in $(seq "$FIRST" 500 "${LAST:-99999}"); do
  until [ -f "$R/model_$it.pt" ]; do sleep 60; [ -d "$R" ] || exit 0; done; sleep 20
  exec 9>"$OUT/gate.lock"; flock 9
  res=$(cd "$REPO"; source scripts/stairs/common.sh; stairs_env "scripts/stairs/$ENVF" >/dev/null 2>&1
    export RHPS1_STAIRS_LVL_FILE= RHPS1_COM_SHIFT_X=-0.037 OSEQ_ONLY=arriere,arriere+gauche
    export RHPS1_ROBUST=$(echo "${RHPS1_ROBUST:-}" | tr ',' '\n' | grep -v '^com$' | paste -sd,)
    .venv/bin/python scripts/stairs/oseq_eval.py "$RUN" "model_$it.pt" "${TAG}_$it" 2>&1 | grep -E "arriere|Error|Traceback")
  exec 9>&-
  bl=$(echo "$res" | grep "arriere+gauche" | grep -oE "tangage min/p5 [-+0-9.]+" | grep -oE "[-+0-9.]+$")
  st=$(echo "$res" | grep -E "^  arriere " | grep -oE "tangage min/p5 [-+0-9.]+" | grep -oE "[-+0-9.]+$")
  [ -z "$bl" ] && { echo "PITCH ${TAG}_$it: no result ($(echo "$res" | head -1))" >> "$OUT/pitch.txt"; continue; }
  best=$(cat "$BEST" 2>/dev/null || echo -99)
  pass=$(awk -v b="$bl" -v t="${PITCH_MIN:--3}" 'BEGIN{print (b >= t) ? 1 : 0}')
  better=$(awk -v b="$bl" -v r="$best" 'BEGIN{print (b > r) ? 1 : 0}')
  verdict=FAIL; [ "$pass" = 1 ] && verdict=PASS
  echo "PITCH ${TAG}_$it: back+left min ${bl} | back min ${st} (CoM -0.037) $verdict, lineage $LIN best $best" >> "$OUT/pitch.txt"
  if [ "$pass" = 1 ] && [ "$better" = 1 ]; then
    echo "$bl" > "$BEST"
    D=$REPO/checkpoints/rhps1/${TAG}_$it
    if [ ! -f "$D/policy.onnx" ]; then
      (cd "$REPO"; mkdir -p "$D"; cp "$R/model_$it.pt" "scripts/stairs/$ENVF" "$D/"
       source scripts/stairs/common.sh; stairs_env "scripts/stairs/$ENVF" >/dev/null 2>&1; export RHPS1_STAIRS_LVL_FILE=
       MJLAB_TASK=Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start MJLAB_LOG_ROOT=logs/rsl_rl/rhps1_tracking CUDA_VISIBLE_DEVICES="" \
         .venv/bin/python scripts/tools/export_onnx.py "$RUN" "model_$it.pt" "$D/policy.onnx" 2>&1 | grep -E "wrote|Error"
       echo "$TAG $it (run $RUN), recipe $ENVF. Pitch gate: back+left min $bl, back min $st deg (CoM -0.037)." > "$D/README.md"
       git pull -q --rebase origin main; git add "$D" && git commit -qm "Checkpoints: $TAG $it with ONNX (pitch gate $bl)" && git push -q origin main
       echo "PITCH ${TAG}_$it: ONNX pushed $(git log --oneline -1)" >> "$OUT/pitch.txt")
    fi
  fi
done
