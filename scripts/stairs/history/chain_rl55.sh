#!/usr/bin/env bash
# chain_rl55.sh: rl54 + single support capped near the BWC's (swing p90 0.62, max 0.64 s): SWING_MAX, AIR_WIN high end and
# STATIC_AIR 1.2 -> ${SW:-0.7} s, from rl54's last checkpoint. rl54 40000 (ftrace 0.5-1.4 cm): every fall = a ~1-1.2 s single
# support, body pitching forward over the stance foot, swing leg left behind and raised (no catch step), stance knee then
# saturates; the air/cadence/static terms paid swings up to 1.2 s.
# Was chain_rl53.sh: rl52 with the loaded-knee cost at W -100 (was -30), from rl52 model_37750. rl52 37500 (500 its at -30, cost
# Was chain_rl52.sh: rl51 + loaded-knee flexion cost (W_KNEE -30, allowance 40 deg + 2 deg/cm), from rl51 model_37000
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
XTRA="RHPS1_STAIRS_W_AIR=1 RHPS1_STAIRS_W_TD=-4 RHPS1_STAIRS_W_IMP=-6 RHPS1_STAIRS_LAND_SHAPE=lin RHPS1_STAIRS_LAND_CAP=3 RHPS1_STAIRS_LAND_AIR=0.03 RHPS1_STAIRS_LAND_RAMP=72000 RHPS1_STAIRS_W_PROG=60 RHPS1_STAIRS_W_STALL=-20 RHPS1_STAIRS_MID_DESC_P=0.5 RHPS1_STAIRS_STALL_S=2 RHPS1_STAIRS_W_TRUNK=-120 RHPS1_STAIRS_TRUNK_DEAD=5 RHPS1_STAIRS_W_VINST=-5 RHPS1_STAIRS_W_VZ=-4 RHPS1_STAIRS_W_TRQ=${W_TRQ:--6} RHPS1_STAIRS_TRQ_FRAC=0.9 RHPS1_STAIRS_TRQ_SHAPE=lin RHPS1_STAIRS_SAT_T=0.4 RHPS1_STAIRS_SAT_T0=0 RHPS1_STAIRS_SAT_RAMP=0 RHPS1_STAIRS_W_KNEE=${W_KNEE:--100} RHPS1_STAIRS_LVL_MAX=${LVL_MAX:-6} RHPS1_STAIRS_W_FCLEAR=-10 RHPS1_STAIRS_W_CLEAR=6 RHPS1_STAIRS_CLEAR_FWD=1 RHPS1_STAIRS_W_TREAD=10 RHPS1_STAIRS_ROWS=20 RHPS1_STAIRS_ARRIVE_FEET=1 RHPS1_STAIRS_W_FALL=-200 RHPS1_STAIRS_FALL_S0=0.15 RHPS1_STAIRS_FCLEAR_RAMP=1 RHPS1_STAIRS_FOOTHOLD_REL=1 RHPS1_STAIRS_STATIC_AIR=${SW:-0.7} RHPS1_STAIRS_W_FOOTHOLD=-10 RHPS1_STAIRS_W_CAD=20 RHPS1_STAIRS_CAD_C=0.2 RHPS1_STAIRS_LAND_RAMP_OFF=90000 RHPS1_STAIRS_UP_LOAD=0.15 RHPS1_STAIRS_STOP_DS=1 RHPS1_STAIRS_PAUSE_P=0.5 RHPS1_STAIRS_SETTLE_T=0.3 RHPS1_STAIRS_SETTLE_V=0.12 RHPS1_STAIRS_W_ARRIVAL=50 RHPS1_STAIRS_ARRIVE_LOADED=1 RHPS1_STAIRS_W_STOP=10 RHPS1_STAIRS_ONE_LEG_T=2.0 RHPS1_STAIRS_INIT_LEVEL=7 RHPS1_STAIRS_SWING_MAX=${SW:-0.7} RHPS1_STAIRS_W_LSWING=-20 RHPS1_STAIRS_T_STEP=1.6 RHPS1_STAIRS_CAD_RAMP=72000 RHPS1_STAIRS_AIR_WIN=0.3,${SW:-0.7} RHPS1_STAIRS_AIR_WIN0=0.05,0.6 RHPS1_STAIRS_W_STANCE=-8 RHPS1_STAIRS_SCAN_FWD=0.3 RHPS1_SCAN_ENC=128,32 RHPS1_STAIRS_MODE_FIXED=1 RHPS1_STAIRS_REPLAY_P=0.2 RHPS1_STAIRS_W_BWC_JV=-8 RHPS1_STAIRS_W_BWC_ANKLE=-2 RHPS1_STAIRS_W_BWC_SWING=-20 RHPS1_STAIRS_BWC_RAMP=0 RHPS1_CYL_LIMITS=1 RHPS1_BOUND_W=1e-2 RHPS1_BOUND=10 RHPS1_KL_CLIP=10 RHPS1_STAIRS_DESC_EDGE=0.05,0.5 RHPS1_STAIRS_STALL_DESC_X=2 RHPS1_STAIRS_LVL_FILE=$S/rhps1_runs/rl55_lvl.pt ${OVERSPEED-RHPS1_STAIRS_W_OVERSPEED=-30 RHPS1_STAIRS_VMAX=0.15}"; GL="$GL --agent.algorithm.entropy-coef 0.0 --agent.clip-actions 12"
last() { ls logs/rsl_rl/rhps1_tracking/$1 | grep -E "^model_[0-9]+\.pt$" | sort -t_ -k2 -n | tail -1; }

R54=2026-11-14_05-51-13
L=$(last $R54); N=$(echo $L | tr -dc 0-9)
for p in $(cat $S/rhps1_runs/rl54.tpid) $(cat $S/rhps1_runs/rl54b.tpid 2>/dev/null); do grep -q train /proc/$p/cmdline 2>/dev/null && kill $p; done
for p in $(cat $S/rhps1_runs/rl54.tpid) $(cat $S/rhps1_runs/rl54b.tpid 2>/dev/null); do while [ -d /proc/$p ]; do sleep 5; done; done
sleep 10
L=$(last $R54); N=$(echo $L | tr -dc 0-9)
cp $S/rhps1_runs/rl54_lvl.pt $S/rhps1_runs/rl55_lvl.pt
echo "rl54 arrete a $N pour rl55 (SWING_MAX ${SW:-0.7}) $(date)" >> $B
stage rl55 $R54 $L 6000 $((N + 500 - N % 500)) RL55
R55=$(cat $S/rhps1_runs/rl55.run); L=$(last $R55); N=$(echo $L | tr -dc 0-9)
stage rl55b $R55 $L 6000 $((N + 500 - N % 500)) RL55B
echo "chain_rl55 fini $(date)" >> $B
