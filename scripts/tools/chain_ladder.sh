#!/usr/bin/env bash
# Wait for the running p0 rung to reach TARGET, stop it, then climb the ladder.
# Only one 4096-env job fits on the 2080 Ti, hence the handoff rather than both.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
LOG=logs/ablation_p0_v2.log
TARGET=${TARGET:-700}
PAT="bin/train Mjlab-Velocity-Flat-RHPS1"

it() { grep -a "Learning iteration" "$LOG" 2>/dev/null | tail -1 |
       grep -oE '[0-9]+/15000' | cut -d/ -f1; }

while true; do
  n=$(it)
  [ -n "${n:-}" ] && [ "$n" -ge "$TARGET" ] && break
  pgrep -f "$PAT" >/dev/null || { echo "$(date -Is) p0_v2 gone at it=${n:-?}"; break; }
  sleep 60
done
echo "$(date -Is) p0_v2 reached it=$(it), stopping it"

pkill -f "$PAT"
for _ in $(seq 1 60); do pgrep -f "$PAT" >/dev/null || break; sleep 2; done
pkill -9 -f "$PAT" 2>/dev/null
sleep 30

echo "$(date -Is) starting rungs 1-8"
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand \
  p0+rand+obs \
  p0+rand+obs+knee \
  p0+rand+obs+knee+feet \
  p0+rand+obs+knee+feet+prox \
  p0+rand+obs+knee+feet+prox+pose \
  p0+rand+obs+knee+feet+prox+pose+mirror \
  p0+rand+obs+knee+feet+prox+pose+mirror+static
