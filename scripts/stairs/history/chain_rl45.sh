#!/usr/bin/env bash
# chain_rl45.sh: rl44 (rl43 + action-mean bound) resumed at 18000, plus the adaptive LR's KL on clipped means:
# moves past the clip kept the KL above target and the LR at its 1e-5 floor from rl40 on (no learning).
cd /home/lmoussafir/mjlab-rhps1
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
RLV=$(grep -o "RHPS1_STAIRS_[A-Z_]*=[^ ]*" $S/stairs/rl_env.sh)
GL="--agent.algorithm.gamma 0.9975 --agent.algorithm.lam 0.987 --agent.algorithm.entropy-coef 0.0"
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
XTRA="RHPS1_STAIRS_W_AIR=1 RHPS1_STAIRS_W_TD=-4 RHPS1_STAIRS_W_IMP=-6 RHPS1_STAIRS_LAND_SHAPE=lin RHPS1_STAIRS_LAND_CAP=3 RHPS1_STAIRS_LAND_AIR=0.03 RHPS1_STAIRS_LAND_RAMP=72000 RHPS1_STAIRS_W_PROG=60 RHPS1_STAIRS_W_STALL=-20 RHPS1_STAIRS_MID_DESC_P=0.5 RHPS1_STAIRS_STALL_S=2 RHPS1_STAIRS_W_TRUNK=-120 RHPS1_STAIRS_TRUNK_DEAD=5 RHPS1_STAIRS_W_VINST=-5 RHPS1_STAIRS_W_VZ=-4 RHPS1_STAIRS_W_TRQ=-2 RHPS1_STAIRS_W_FCLEAR=-10 RHPS1_STAIRS_W_CLEAR=6 RHPS1_STAIRS_CLEAR_FWD=1 RHPS1_STAIRS_W_TREAD=10 RHPS1_STAIRS_ROWS=20 RHPS1_STAIRS_ARRIVE_FEET=1 RHPS1_STAIRS_W_FALL=-200 RHPS1_STAIRS_FALL_S0=0.15 RHPS1_STAIRS_FCLEAR_RAMP=1 RHPS1_STAIRS_FOOTHOLD_REL=1 RHPS1_STAIRS_STATIC_AIR=1.2 RHPS1_STAIRS_W_FOOTHOLD=-10 RHPS1_STAIRS_W_CAD=20 RHPS1_STAIRS_CAD_C=0.2 RHPS1_STAIRS_LAND_RAMP_OFF=90000 RHPS1_STAIRS_UP_LOAD=0.15 RHPS1_STAIRS_STOP_DS=1 RHPS1_STAIRS_PAUSE_P=0.5 RHPS1_STAIRS_SETTLE_T=0.3 RHPS1_STAIRS_SETTLE_V=0.12 RHPS1_STAIRS_W_ARRIVAL=50 RHPS1_STAIRS_ARRIVE_LOADED=1 RHPS1_STAIRS_W_STOP=10 RHPS1_STAIRS_ONE_LEG_T=2.0 RHPS1_STAIRS_INIT_LEVEL=3 RHPS1_STAIRS_SWING_MAX=1.2 RHPS1_STAIRS_W_LSWING=-20 RHPS1_STAIRS_T_STEP=1.6 RHPS1_STAIRS_CAD_RAMP=72000 RHPS1_STAIRS_AIR_WIN=0.3,1.2 RHPS1_STAIRS_AIR_WIN0=0.05,0.6 RHPS1_STAIRS_W_STANCE=-8 RHPS1_STAIRS_SCAN_FWD=0.3 RHPS1_SCAN_ENC=128,32 RHPS1_STAIRS_MODE_FIXED=1 RHPS1_STAIRS_REPLAY_P=0.2 RHPS1_STAIRS_W_BWC_JV=-8 RHPS1_STAIRS_W_BWC_ANKLE=-2 RHPS1_STAIRS_W_BWC_SWING=-8 RHPS1_CYL_LIMITS=1 RHPS1_BOUND_W=1e-3 RHPS1_BOUND=${BOUND:-10} RHPS1_KL_CLIP=12"; GL="$GL --agent.algorithm.entropy-coef 0.0 --agent.clip-actions 12"
last() { ls logs/rsl_rl/rhps1_tracking/$1 | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1; }
R44=$(cat $S/rhps1_runs/rl44.run)
stage rl45 $R44 model_18000.pt 6000 18500 RL45
R45=$(cat $S/rhps1_runs/rl45.run); L45=$(last $R45); N45=$(echo $L45 | tr -dc 0-9)
stage rl45b $R45 $L45 6000 $((N45 + 500 - N45 % 500)) RL45B
echo "chain_rl45 fini $(date)" >> $B
