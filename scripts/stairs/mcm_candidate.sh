#!/usr/bin/env bash
# Robot-candidate bench (tardis 2026-11-27): slot 18 rests at IMU +4.1 / CoM +4.6 cm; gimpCT 10000 +2.4 / +4.7, diagonal -3.0, straight back -2.6.
# Rest posture: mcm_rest_com.py <log> <label> (CoM ahead of the ankles), mcm_restpose.py <log> <label> (q - q0 pre-arm vs rest).
# mcm_candidate.sh <name> <onnx> [slot]: rest IMU pitch, back+left diagonal (KI 0.02/2.0, CoM -0.037), four axes
N=$1; O=$2; SL=${3:-25}; P=$(dirname "$(readlink -f "$0")")
cd ~/mjlab-rhps1/scripts/stairs && source common.sh && export RLQP_BOXES="30 30.1 30 30.1 0.001" RLQP_BENCH_VEL_DUR_S=8 VST=2
L=$(BENCH_VEL="0.15 0 0" RLQP_BENCH_POLICY_INDEX=$SL RLQP_BENCH_POLICY_PATH=$O bash headless.sh 0 0.15 20 2>/dev/null | tail -1)
python3 - "$L" "$N" <<'EOF' 2>&1 | grep -v PyQt
import sys, mc_log_ui, numpy as np
l=mc_log_ui.read_log(sys.argv[1]); t=np.asarray(l['t']); g=lambda k: np.asarray(l[k])
ax,ay,az=[g('Accelerometer_linearAcceleration_'+a) for a in 'xyz']
k=40; f=lambda x: np.convolve(x,np.ones(k)/k,mode='same'); ax,az=f(ax),f(az)
pitch=np.degrees(np.arctan2(-ax,az))
vc=g('NewRLQPController_velCmd_x'); on=np.where(np.abs(vc)>1e-3)[0]; t0,t1=t[on[0]],t[on[-1]]
pre=(t>t0-1.5)&(t<t0-0.2); wk=(t>t0+2)&(t<t1); post=(t>t1+3)&(t<t1+6)
print(f"{sys.argv[2]}: IMU pitch at arming {pitch[pre].mean():+.1f} | walking median {np.median(pitch[wk]):+.1f} | standing after {pitch[post].mean():+.1f} deg  (slot 18: +3.4 / +0.2 / +4.1; slot 24: +3.3 / -0.9 / -6.0; gimpP2 7000: +3.3 / -1.0 / -2.0)")
EOF
rm -f $L
for ki in 0.02 2.0; do L=$(MC_MUJOCO_RHPS1_KI_ANKLE=$ki MC_MUJOCO_RHPS1_CHEST_COM_X=-0.037 RLQP_BENCH_VEL_SEQ="3 0.15 0 0; 4 0 0 0; 3 -0.28 0.15 0; 6 0 0 0" RLQP_BENCH_POLICY_INDEX=$SL RLQP_BENCH_POLICY_PATH=$O bash headless.sh 0 0.15 24 2>/dev/null | tail -1); python3 $P/mcm_fallcheck.py $L "$N back+left KI $ki com" 2>&1 | grep -v PyQt; rm -f $L; done
LOGS=(); for v in "0.15 0 0" "-0.15 0 0" "0 0.15 0" "0 0 0.3"; do L=$(BENCH_VEL="$v" RLQP_BENCH_POLICY_INDEX=$SL RLQP_BENCH_POLICY_PATH=$O bash headless.sh 0 0.15 20 2>/dev/null | tail -1); LOGS+=($L); done
cd $REPO && .venv/bin/python scripts/stairs/mcm_metrics.py "${LOGS[@]}" | sed '2s/^/fwd  /;3s/^/back /;4s/^/left /;5s/^/yaw+ /'
Q=(fwd back); for i in 0 1; do .venv/bin/python scripts/stairs/mcm_quality.py ${LOGS[$i]} "$N ${Q[$i]}" | grep -E "land|rest|slip"; done; rm -f "${LOGS[@]}"
