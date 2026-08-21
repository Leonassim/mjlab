#!/usr/bin/env bash
# Watchdog for an unattended training run. Exits -- returning control -- on the
# first of: milestone reached, torque guard tripped, or the trainer dying.
#
# Never uses `pgrep -f`: the pattern matches this script's own cmdline, which
# has deadlocked a gate here before. Tracks the trainer by pid.
#
#   MILESTONE   iteration at which to hand back for a report   (default 300)
#   TORQUE_MAX  torque_limit_ratio_mean ceiling, 2 samples     (default 0.45)
#   POLL        seconds between samples                        (default 300)
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
MILESTONE=${MILESTONE:-300}
TORQUE_MAX=${TORQUE_MAX:-0.45}
POLL=${POLL:-300}
TSV=logs/watch_$(date +%m%d_%H%M).tsv
START=$(date +%s)

PID=""
for _ in $(seq 1 40); do
  PID=$(ps -eo pid,cmd | awk '/venv\/bin\/train/ && !/awk/ {print $1; exit}')
  [ -n "$PID" ] && break
  sleep 30
done
[ -z "$PID" ] && { echo "GUARD: trainer never appeared"; exit 4; }

# Wait for the run directory this trainer creates. Taking the newest directory
# outright once latched onto the previous run, read its final values and
# declared the milestone reached in one sample. Run dirs are named for their
# start time, so require one at or after ours.
RUN=${RUN_DIR:-}          # set it explicitly when attaching to a live run
for _ in $(seq 1 60); do
  [ -n "$RUN" ] && break
  CAND=$(ls -dt logs/rsl_rl/rhps1_velocity/*/ 2>/dev/null | head -1)
  if [ -n "$CAND" ]; then
    NAME=$(basename "$CAND")
    T=$(date -d "$(echo "$NAME" | sed 's/_/ /; s/-/:/3; s/-/:/3')" +%s 2>/dev/null || echo 0)
    [ "$T" -ge "$((START - 120))" ] && { RUN="$CAND"; break; }
  fi
  sleep 20
done
[ -z "$RUN" ] && { echo "GUARD: no run directory newer than this watch"; exit 5; }
echo "watching pid $PID  run $RUN  milestone $MILESTONE  torque_max $TORQUE_MAX"
printf 'it\tstep_len\tpeak_h\tair_t\ttorque\ttq_max\tsway\tvel_err\tfell\taction\n' > "$TSV"

STRIKES=0
while :; do
  sleep "$POLL"
  if ! kill -0 "$PID" 2>/dev/null; then echo "GUARD: trainer $PID exited"; exit 2; fi
  ROW=$(.venv/bin/python scripts/tools/watch_probe.py "$RUN" 2>/dev/null)
  [ -z "$ROW" ] && continue
  echo "$ROW" >> "$TSV"; echo "$ROW"
  IT=$(echo "$ROW" | cut -f1); TQ=$(echo "$ROW" | cut -f5)
  if [ "${PREV_IT:-0}" = "$IT" ]; then
    echo "GUARD: iteration stuck at $IT -- wrong run directory or trainer hung"; exit 6
  fi
  PREV_IT=$IT
  if awk -v a="$TQ" -v b="$TORQUE_MAX" 'BEGIN{exit !(a+0>b+0)}'; then
    STRIKES=$((STRIKES+1))
    [ "$STRIKES" -ge 2 ] && { echo "GUARD: torque $TQ over $TORQUE_MAX twice"; exit 3; }
  else STRIKES=0; fi
  if [ "${IT%.*}" -ge "$MILESTONE" ] 2>/dev/null; then echo "MILESTONE $IT"; exit 0; fi
done
