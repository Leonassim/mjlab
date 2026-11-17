#!/usr/bin/env bash
# headless.sh <h> <vx> <secs> [tread]: stairs slot (22) in mc_mujoco without GUI, auto-armed at 3 s, command vx from VST s
# after arming (default 2). Prints the mc_rtc log path. RLQP_BENCH_POLICY_PATH / RLQP_BENCH_POLICY_STEP pass through.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for _ in $(seq 60); do pgrep -x mc_mujoco >/dev/null || break; sleep 1; done  # previous run still closing
pgrep -x mc_mujoco >/dev/null && { echo "mc_mujoco already running: bench skipped"; exit 3; }
H=$1; VX=$2; SECS=$3; T=${4:-0.27}
export RLQP_BENCH_AUTO_ARM=3.0 RLQP_BENCH_VELOCITY="$VX 0 0" RLQP_BENCH_VEL_START_S=${VST:-2.0}
touch "$RUNS/headless.start"
# timeout: the run ends even when the group kill below misses it (setsid may fork, then $! is not the group).
setsid timeout -k 10 $((SECS + 360)) "$RLC/scripts/mc_mujoco_stairs.sh" $H $T -- --sync --without-visualization --without-mc-rtc-gui > "$RUNS/headless.log" 2>&1 &
PGID=$!
for _ in $(seq 1 60); do
  sleep 5
  find ~/logs/mc_rtc -name "mc-control-NewRLQPController-2*.bin" -newer "$RUNS/headless.start" | grep -q . && break
  pgrep -x mc_mujoco >/dev/null || { echo "mc_mujoco died before init"; break; }
done
sleep "$SECS"
kill -TERM -$PGID 2>/dev/null
for _ in $(seq 1 20); do kill -0 $PGID 2>/dev/null || break; sleep 1; done
kill -KILL -$PGID 2>/dev/null; sleep 1
find ~/logs/mc_rtc -name "mc-control-NewRLQPController-2*.bin" -newer "$RUNS/headless.start" | sort | tail -1
