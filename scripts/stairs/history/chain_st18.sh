#!/usr/bin/env bash
# chain_st18.sh: st17 climb-only (Leo), foothold also on the landing swing foot (hit heights, last 5 cm), weight -5: post-contact -10 taught edge avoidance.
cd /home/lmoussafir/mjlab-rhps1
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
P=$(EXTRA_TRAIN="--agent.algorithm.gamma 0.9975 --agent.algorithm.lam 0.987" NUM_ENVS=${NUM_ENVS:-3584} $S/stairs/st.sh st18 4000 RHPS1_STAIRS_NJMAX=${NJMAX:-0} RHPS1_STAIRS_NCONMAX=${NCONMAX:-0} RHPS1_STAIRS_H=0.005,0.18 RHPS1_STAIRS_INIT_LEVEL=0 RHPS1_STAIRS_EPISODE_S=16 RHPS1_STAIRS_W_TEACHER=-10 RHPS1_STAIRS_TEACHER_MASK=flat RHPS1_STAIRS_W_ARRIVAL=0 RHPS1_STAIRS_W_CLEAR=2 RHPS1_STAIRS_W_STATIC=0 RHPS1_STAIRS_SPEED=0.085 RHPS1_STAIRS_SCAN_NOISE=0 RHPS1_STAIRS_W_ALIVE=8 RHPS1_STAIRS_W_DS=1 RHPS1_STAIRS_VEL_TAU=1.0 RHPS1_STAIRS_W_FOOTHOLD=-5 RHPS1_STAIRS_FOOTHOLD_PRE=1 RHPS1_STAIRS_CLIMB_FRAC=1.0)
echo $P > $S/rhps1_runs/st18.tpid
until grep -q -E "Learning iteration|Traceback|Error" $S/rhps1_runs/st18.log; do sleep 5; done
if grep -q -E "Traceback|OutOfMemory" $S/rhps1_runs/st18.log; then echo "st18 ECHEC au lancement" >> $B; exit 1; fi
R=$(basename $(grep -m1 -o "logs/rsl_rl/rhps1_tracking/[0-9_-]*" $S/rhps1_runs/st18.log)); echo $R > $S/rhps1_runs/st18.run
echo "st18 lance PID $P -> $R $(date)" >> $B
D=logs/rsl_rl/rhps1_tracking/$R
for it in $(seq 13750 250 17497); do
  until [ -f $D/model_$it.pt ] || ! [ -d /proc/$P ]; do sleep 30; done
  [ -f $D/model_$it.pt ] || it=$(ls $D | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1 | tr -dc 0-9)
  sleep 20
  (source $S/stairs/env.sh; export RHPS1_STAIRS_H=0.005,0.18 RHPS1_STAIRS_SPEED=0.085 RHPS1_STAIRS_SCAN_NOISE=0 RHPS1_STAIRS_VEL_TAU=1.0 RHPS1_STAIRS_W_FOOTHOLD=-5 RHPS1_STAIRS_FOOTHOLD_PRE=1; BENCH_PER=2 BENCH_FLAT=8 nice -n 5 timeout 3000 .venv/bin/python $S/stairs/stairs_bench.py $R model_$it.pt ST18_$it 2>/dev/null | sed -n '/^STAIRS/,$p') >> $B
  [ -d /proc/$P ] || break
done
echo "chain_st18 fini $(date)" >> $B
