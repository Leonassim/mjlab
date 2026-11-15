#!/usr/bin/env bash
# chain_s50.sh: 50 Hz policy from scratch (Leo 2026-11-14). Decimation 8 (physics 2.5 ms), gamma 0.99 / lam 0.95 (same 2 s /
# 0.33 s horizons), action_rate / 16 (same cost for the same motion; measured -0.05/s det, -0.46/s with noise on ap6 45500),
# landing peak over substeps, 50 ms landing-speed window. Flat tiles run the RL task (20 %), 10 % of episodes get no command,
# stairs approach 0.3-2 m at +-30 deg ramped in, stops on the flight, regularizers / cadence / BWC envelope / saturation
# CaT ramped over 72000 env steps (3000 its), cheap falls first (FALL_S0 0.15), curriculum from 0.5 cm capped at 6 cm.
cd /home/lmoussafir/mjlab-rhps1
# Benches launched by stage() inherit it: a 50 Hz policy must be judged at 50 Hz.
export RHPS1_DECIMATION=8
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
RLV=$(grep -o "RHPS1_STAIRS_[A-Z_]*=[^ ]*" $S/stairs/rl_env.sh)
GL="--agent.algorithm.gamma 0.99 --agent.algorithm.lam 0.95 --agent.algorithm.entropy-coef 0.002"
stage() {  # stage <tag> <load_run> <load_ck> <iters> <first_bench> <label>
  local TAG=$1 LR=$2 LC=$3 N=$4 FB=$5 LB=$6
  local P=$(LOAD_RUN=$LR LOAD_CK=$LC EXTRA_TRAIN="$GL" NUM_ENVS=${NUM_ENVS:-3584} $S/stairs/st.sh $TAG $N RHPS1_ANKLE_VIB_W=-4 $RLV $XTRA)
  echo $P > $S/rhps1_runs/$TAG.tpid
  until grep -q -E "Learning iteration|Traceback|Error" $S/rhps1_runs/$TAG.log; do sleep 5; done
  if grep -q -E "Traceback|OutOfMemory" $S/rhps1_runs/$TAG.log; then echo "$TAG ECHEC au lancement" >> $B; return 1; fi
  local R=$(basename $(grep -m1 -o "logs/rsl_rl/rhps1_tracking/[0-9_-]*" $S/rhps1_runs/$TAG.log)); echo $R > $S/rhps1_runs/$TAG.run
  echo "$TAG lance PID $P -> $R $(date)" >> $B
  local D=logs/rsl_rl/rhps1_tracking/$R it
  for it in $(seq $FB 500 99999); do
    until [ -f $D/model_$it.pt ] || ! [ -d /proc/$P ]; do sleep 30; done
    [ -f $D/model_$it.pt ] || it=$(ls $D | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1 | tr -dc 0-9)
    sleep 20
    (source $S/stairs/env.sh; source $S/stairs/rl_env.sh; export RHPS1_CYL_LIMITS=1 CLIP_ACTIONS=12 RHPS1_STAIRS_SCAN_FWD=0.3 RHPS1_SCAN_ENC=128,32 RHPS1_ANKLE_VIB_W=-4 RHPS1_STAIRS_ROWS=20 RHPS1_STAIRS_ARRIVE_FEET=1 BENCH_LEVELS=0,2,4,8,19; BENCH_NO_OVERRIDE=1 BENCH_FLAT=0 BENCH_T=24 BENCH_PER=2 nice -n 5 timeout 3000 .venv/bin/python $S/stairs/stairs_bench.py $R model_$it.pt ${LB}_$it 2>/dev/null | sed -n '/^STAIRS/,$p') >> $B
    (source $S/stairs/env.sh; source $S/stairs/rl_env.sh; export RHPS1_CYL_LIMITS=1 CLIP_ACTIONS=12 RHPS1_STAIRS_SCAN_FWD=0.3 RHPS1_SCAN_ENC=128,32 RHPS1_STAIRS_ROWS=20 RHPS1_STAIRS_ARRIVE_FEET=1 BENCH_LEVELS=0,4,8; BENCH_PER=4 nice -n 5 timeout 2700 .venv/bin/python $S/stairs/stairs_gait.py $R model_$it.pt ${LB}_$it 2>/dev/null | sed -n "/^GAIT/,\$p") >> $B
    [ -d /proc/$P ] || break
  done
  echo "$TAG fini $(date)" >> $B
}
XTRA="RHPS1_STAIRS_W_AIR=1 RHPS1_STAIRS_W_TD=-4 RHPS1_STAIRS_W_IMP=-6 RHPS1_STAIRS_LAND_SHAPE=lin RHPS1_STAIRS_LAND_CAP=3 RHPS1_STAIRS_LAND_AIR=0.03 RHPS1_STAIRS_LAND_RAMP=72000 RHPS1_STAIRS_W_PROG=60 RHPS1_STAIRS_W_STALL=-20 RHPS1_STAIRS_MID_DESC_P=0.5 RHPS1_STAIRS_STALL_S=2 RHPS1_STAIRS_W_TRUNK=-120 RHPS1_STAIRS_TRUNK_DEAD=5 RHPS1_STAIRS_W_VINST=-5 RHPS1_STAIRS_W_VZ=-4 RHPS1_STAIRS_W_TRQ=${W_TRQ:--6} RHPS1_STAIRS_TRQ_FRAC=0.9 RHPS1_STAIRS_TRQ_SHAPE=lin RHPS1_STAIRS_SAT_T=0.4 RHPS1_STAIRS_SAT_T0=0 RHPS1_STAIRS_SAT_RAMP=0 RHPS1_STAIRS_W_KNEE=${W_KNEE:--100} RHPS1_STAIRS_LVL_MAX=${LVL_MAX:-6} RHPS1_STAIRS_W_FCLEAR=-10 RHPS1_STAIRS_W_CLEAR=6 RHPS1_STAIRS_CLEAR_FWD=1 RHPS1_STAIRS_W_TREAD=10 RHPS1_STAIRS_ROWS=20 RHPS1_STAIRS_ARRIVE_FEET=1 RHPS1_STAIRS_W_FALL=-200 RHPS1_STAIRS_FALL_S0=0.15 RHPS1_STAIRS_FCLEAR_RAMP=1 RHPS1_STAIRS_FOOTHOLD_REL=1 RHPS1_STAIRS_STATIC_AIR=1.2 RHPS1_STAIRS_W_FOOTHOLD=-10 RHPS1_STAIRS_W_CAD=20 RHPS1_STAIRS_CAD_C=0.2 RHPS1_STAIRS_LAND_RAMP_OFF=90000 RHPS1_STAIRS_UP_LOAD=0.15 RHPS1_STAIRS_STOP_DS=1 RHPS1_STAIRS_PAUSE_P=0.5 RHPS1_STAIRS_SETTLE_T=0.3 RHPS1_STAIRS_SETTLE_V=0.12 RHPS1_STAIRS_W_ARRIVAL=50 RHPS1_STAIRS_ARRIVE_LOADED=1 RHPS1_STAIRS_W_STOP=10 RHPS1_STAIRS_ONE_LEG_T=2.0 RHPS1_STAIRS_INIT_LEVEL=7 RHPS1_STAIRS_SWING_MAX=1.2 RHPS1_STAIRS_W_LSWING=-20 RHPS1_STAIRS_T_STEP=1.6 RHPS1_STAIRS_CAD_RAMP=72000 RHPS1_STAIRS_AIR_WIN=0.3,1.2 RHPS1_STAIRS_AIR_WIN0=0.05,0.6 RHPS1_STAIRS_W_STANCE=-8 RHPS1_STAIRS_SCAN_FWD=0.3 RHPS1_SCAN_ENC=128,32 RHPS1_STAIRS_MODE_FIXED=1 RHPS1_STAIRS_REPLAY_P=0.2 RHPS1_STAIRS_W_BWC_JV=-8 RHPS1_STAIRS_W_BWC_ANKLE=-2 RHPS1_STAIRS_W_BWC_SWING=-20 RHPS1_STAIRS_BWC_RAMP=0 RHPS1_CYL_LIMITS=1 RHPS1_BOUND_W=1e-2 RHPS1_BOUND=10 RHPS1_KL_CLIP=10 RHPS1_STAIRS_DESC_EDGE=0.05,0.5 RHPS1_STAIRS_STALL_DESC_X=2 RHPS1_STAIRS_LVL_FILE=$S/rhps1_runs/s50_lvl.pt ${OVERSPEED-RHPS1_STAIRS_W_OVERSPEED=-30 RHPS1_STAIRS_VMAX=0.15} RHPS1_STAIRS_GAP=0.5,1.5 RHPS1_STAIRS_YAW_SPAWN=30 RHPS1_STAIRS_ARRIVE_DESC=0.8 RHPS1_STAIRS_DESC_EDGE=0.2,0.8 RHPS1_STAIRS_EPISODE_S=32 RHPS1_STAIRS_INIT_LEVEL=4 RHPS1_STAIRS_STAIR_PAUSE_P=0.5 RHPS1_STAIRS_W_SAMETREAD=2 RHPS1_STAIRS_APPROACH_RAMP=72000 RHPS1_STAIRS_W_HEADERR=-3 RHPS1_DECIMATION=8 RHPS1_STAIRS_FPEAK_SUB=1 RHPS1_ACTION_RATE_W=-0.00625 RHPS1_STAIRS_FRAC=0.8 RHPS1_STAIRS_FLAT_RL=1 RHPS1_STAIRS_STAND_P=0.1 RHPS1_STAIRS_LAND_RAMP=72000 RHPS1_STAIRS_LAND_RAMP_OFF=0 RHPS1_STAIRS_CAD_RAMP=72000 RHPS1_STAIRS_BWC_RAMP=72000 RHPS1_STAIRS_SAT_T0=1.0 RHPS1_STAIRS_SAT_RAMP=72000 RHPS1_STAIRS_APPROACH_RAMP=72000 RHPS1_STAIRS_APPROACH_OFF=0 RHPS1_STAIRS_INIT_LEVEL=0 RHPS1_STAIRS_GAP=0.3,2.0"; GL="$GL --agent.clip-actions 12 --video-length 150"
last() { ls logs/rsl_rl/rhps1_tracking/$1 | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1; }

for p in $(cat $S/rhps1_runs/ap7.tpid); do grep -q train /proc/$p/cmdline 2>/dev/null && kill $p; done
for p in $(cat $S/rhps1_runs/ap7.tpid); do while [ -d /proc/$p ]; do sleep 5; done; done
sleep 10
rm -f $S/rhps1_runs/s50_lvl.pt
echo "ap7 arrete pour s50 : 50 Hz depuis zero (plat RL 20 %, debout 10 %, escaliers 80 %, approche 0.3-2 m) $(date)" >> $B
stage s50 - - 20000 1000 S50
RA=$(cat $S/rhps1_runs/s50.run); L=$(last $RA); N=$(echo $L | tr -dc 0-9)
stage s50b $RA $L 20000 $((N + 500 - N % 500)) S50B
echo "chain_s50 fini $(date)" >> $B
