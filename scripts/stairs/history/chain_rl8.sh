#!/usr/bin/env bash
# chain_rl8.sh: rl7 at 2750 with linear landing costs capped at 10x (W -0.5 each, ramp 500 its): rl7 stomped at 2 m/s, 4 kN p90; then rl2b.
cd /home/lmoussafir/mjlab-rhps1
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
RLV=$(grep -o "RHPS1_STAIRS_[A-Z_]*=[^ ]*" $S/stairs/rl_env.sh)
GL="--agent.algorithm.gamma 0.9975 --agent.algorithm.lam 0.987"
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
    (source $S/stairs/env.sh; source $S/stairs/rl_env.sh; export RHPS1_ANKLE_VIB_W=-4; BENCH_NO_OVERRIDE=1 BENCH_FLAT=0 BENCH_T=24 BENCH_PER=2 nice -n 5 timeout 3000 .venv/bin/python $S/stairs/stairs_bench.py $R model_$it.pt ${LB}_$it 2>/dev/null | sed -n '/^STAIRS/,$p') >> $B
    (source $S/stairs/env.sh; source $S/stairs/rl_env.sh; nice -n 5 timeout 1800 .venv/bin/python $S/stairs/stairs_impact.py $R model_$it.pt ${LB}_$it 2>/dev/null | sed -n "/^IMPACT/,\$p") >> $B
    [ -d /proc/$P ] || break
  done
  echo "$TAG fini $(date)" >> $B
}
XTRA="RHPS1_STAIRS_W_AIR=1 RHPS1_STAIRS_W_TD=-0.5 RHPS1_STAIRS_W_IMP=-0.5 RHPS1_STAIRS_LAND_SHAPE=lin RHPS1_STAIRS_LAND_CAP=10 RHPS1_STAIRS_LAND_RAMP=12000 RHPS1_STAIRS_W_FALL=-200"; GL="$GL --agent.algorithm.entropy-coef 0.001"
stage rl8 2026-11-09_18-04-59 model_2750.pt 12250 3000 RL8
XTRA=""; GL="--agent.algorithm.gamma 0.9975 --agent.algorithm.lam 0.987"
stage rl2b 2026-11-06_stairs_init model_13498.pt 6000 14000 RL2B
echo "chain_rl8 fini $(date)" >> $B
