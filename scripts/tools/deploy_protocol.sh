#!/usr/bin/env bash
# Deployment protocol for one tracking checkpoint: the same scenarios in mjlab and
# in mc_mujoco, side by side, against the BWC.
#
#   scripts/tools/deploy_protocol.sh <run> <model_XXXX.pt> [cmd_tau=0.3] [delay_ms=30]
#
# 1. export the ONNX, check its scales against slot 4;
# 2. mjlab: delay_probe (delay_ms and 45 ms) and flip_probe (joystick push);
# 3. mc_mujoco, slot 4 with this ONNX, target delayed by delay_ms and the command
#    filtered by cmd_tau (RLQP_BENCH_*, mc_mujoco only): stand, walk, joystick
#    push 6 s after arming. The installed ONNX is restored on exit;
# 4. table mjlab / mc_mujoco / BWC, flagging every mjlab-mc_mujoco gap.
#
# Slot 4's arm_posture was measured on it50000; a new policy arms into it anyway.
# 30 ms is the real robot's lag beyond the QP (2026-09-30 log), so mc_mujoco at
# 30 ms is the closest stand-in for the robot we have.
set -u
RUN=$1; CK=$2; TAU=${3:-0.3}; DL=${4:-30}
cd /home/lmoussafir/mjlab-rhps1 || exit 1
OUT=${SCRATCH:-/tmp}/deploy_protocol/${RUN}_${CK%.pt}
mkdir -p "$OUT"
PLANT=(RHPS1_POSTURE_K=1600 RHPS1_POSTURE_K_HI=0 RHPS1_FOOT_FRICTION=1.0 RHPS1_ACTION_SCALE_MULT=10)
TAUENV=(); BTAU=()
if [ "$TAU" != 0 ]; then TAUENV=(RHPS1_CMD_TAU="$TAU"); BTAU=(RLQP_BENCH_CMD_TAU="$TAU"); fi

echo "== $RUN $CK  (cmd_tau $TAU, retard $DL ms)  -> $OUT"
env CUDA_VISIBLE_DEVICES= "${PLANT[@]}" "${TAUENV[@]}" MJLAB_MOTION_FILE=docs/bwc_motion_v2.npz \
  MJLAB_TASK=Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start MJLAB_LOG_ROOT=logs/rsl_rl/rhps1_tracking \
  timeout 900 uv run python scripts/tools/export_onnx.py "$RUN" "$CK" "$OUT/policy.onnx" 2>&1 | tail -1
uv run python scripts/tools/check_slot_scales.py "$OUT/policy.onnx" 4 2>&1 | tail -1 | tee "$OUT/scales.txt"
grep -q "==" "$OUT/scales.txt" || { echo "echelles differentes du slot 4 : arret"; exit 1; }

echo "-- mjlab"
env CUDA_VISIBLE_DEVICES= "${PLANT[@]}" timeout 1700 uv run python scripts/tools/delay_probe.py \
  "$RUN" "$CK" "$DL" 45 2>&1 | grep -E '^lag' | tee "$OUT/mjlab_delay.txt"
env CUDA_VISIBLE_DEVICES= "${PLANT[@]}" "${TAUENV[@]}" FLIP_DELAY_MS="$DL" timeout 1700 uv run python \
  scripts/tools/flip_probe.py "$RUN" "$CK" 2>&1 | grep -E 's  bassin|avance' | tee "$OUT/mjlab_flip.txt"

echo "-- mc_mujoco"
P=/home/lmoussafir/install/share/NewRLQPController/policy/rhps1_bwc_x10.onnx
cp "$P" "$OUT/slot4_backup.onnx"
trap 'cp "$OUT/slot4_backup.onnx" "$P"; echo "slot 4 restaure ($(md5sum "$P" | cut -c1-8))"' EXIT
cp "$OUT/policy.onnx" "$P"
LOGS=/home/lmoussafir/logs/mc_rtc
for scen in stand:0:15: walk:0.3:15: flip:0.3:16:6; do
  IFS=: read -r name vx secs start <<< "$scen"
  touch "$OUT/.mark"
  START=(); [ -n "$start" ] && START=(RLQP_BENCH_VEL_START_S="$start")
  env RLQP_BENCH_DELAY_MS="$DL" "${BTAU[@]}" "${START[@]}" \
    bash scripts/tools/mc_mujoco_bench.sh 4 "$vx" 0 0 "$secs" 2>&1 | grep -E 'ATTENTION|mort|saute'
  log=$(find "$LOGS" -name 'mc-control-NewRLQPController-2*.bin' -newer "$OUT/.mark" | sort | tail -1)
  [ -z "$log" ] && { echo "$name : aucun log"; continue; }
  mode=$name; [ "$name" = stand ] && mode=walk
  python3 scripts/tools/mc_mujoco_read.py "$mode" "$log" | tee "$OUT/mc_$name.txt"
done

echo "-- comparaison"
python3 - "$OUT" <<'EOF'
import re, sys
out = sys.argv[1]
rd = lambda f: open(f"{out}/{f}").read() if __import__("os").path.exists(f"{out}/{f}") else ""
win = r"(\d\.\d-\d\.\d) s\s+bassin\s+([\d.]+)\s*/\s*[\d.]+ deg/s\s+haut du corps\s+([\d.]+)"
mj = {w: (float(p), float(u)) for w, p, u in re.findall(win, rd("mjlab_flip.txt"))}
mc = {w: (float(p), float(u)) for w, p, u in re.findall(win, rd("mc_flip.txt"))}
bwc = {"0.0-0.3": (0, 0), "0.3-1.0": (0, 0), "1.0-2.3": (8, 8)}
adv = lambda s: (re.findall(r"avance 5 s apres la bascule\s+([-\d.]+)", s) or ["nan"])[0]
flag = lambda a, b: "  <-- ecart" if abs(a - b) > max(3.0, 0.5 * max(a, b)) else ""
print(f"{'joystick, deg/s':22s}{'mjlab':>14s}{'mc_mujoco':>14s}{'BWC':>10s}")
for w in bwc:
  for k, name in ((0, "bassin"), (1, "haut du corps")):
    a = mj.get(w, (float("nan"),) * 2)[k]; b = mc.get(w, (float("nan"),) * 2)[k]
    print(f"{w + ' ' + name:22s}{a:14.1f}{b:14.1f}{bwc[w][k]:10.0f}{flag(a, b)}")
a, b = adv(rd("mjlab_flip.txt")), adv(rd("mc_flip.txt"))
gap = "  <-- ecart" if a != "nan" and b != "nan" and abs(float(a) - float(b)) > 0.1 else ""
print(f"{'avance 5 s (m)':22s}{a:>14s}{b:>14s}{'0.42':>10s}{gap}")
print("retard (mjlab, 16 departs) :")
for l in rd("mjlab_delay.txt").splitlines():
  m = re.search(r"lag\s+(\d+) ms.*fell (\d+)/(\d+)", l)
  if m: print(f"  {m.group(1)} ms : {m.group(2)}/{m.group(3)} chutes")
for n in ("stand", "walk", "flip"):
  s = rd(f"mc_{n}.txt").strip().splitlines()
  print(f"mc_mujoco {n:5s} : {s[-1].strip() if s else 'absent'}")
EOF
