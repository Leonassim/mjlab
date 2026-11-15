#!/usr/bin/env bash
# chain_st21.sh: stairs expert first cut: st18 from slot 20 with 90 % stair tiles, no slot 20 anchor (it exploded in st19/st20), 24 s episodes; bench at 24 s.
cd /home/lmoussafir/mjlab-rhps1
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
P=$(EXTRA_TRAIN="--agent.algorithm.gamma 0.9975 --agent.algorithm.lam 0.987" NUM_ENVS=${NUM_ENVS:-3584} $S/stairs/st.sh st21 4000 RHPS1_STAIRS_NJMAX=${NJMAX:-0} RHPS1_STAIRS_NCONMAX=${NCONMAX:-0} RHPS1_STAIRS_H=0.005,0.18 RHPS1_STAIRS_INIT_LEVEL=0 RHPS1_STAIRS_EPISODE_S=24 RHPS1_STAIRS_W_TEACHER=0 RHPS1_STAIRS_FRAC=0.9 RHPS1_STAIRS_W_ARRIVAL=0 RHPS1_STAIRS_W_CLEAR=2 RHPS1_STAIRS_W_STATIC=0 RHPS1_STAIRS_SPEED=0.085 RHPS1_STAIRS_SCAN_NOISE=0 RHPS1_STAIRS_W_ALIVE=8 RHPS1_STAIRS_W_DS=1 RHPS1_STAIRS_VEL_TAU=1.0 RHPS1_STAIRS_W_FOOTHOLD=-5 RHPS1_STAIRS_FOOTHOLD_PRE=1 RHPS1_STAIRS_CLIMB_FRAC=1.0)
echo $P > $S/rhps1_runs/st21.tpid
until grep -q -E "Learning iteration|Traceback|Error" $S/rhps1_runs/st21.log; do sleep 5; done
if grep -q -E "Traceback|OutOfMemory" $S/rhps1_runs/st21.log; then echo "st21 ECHEC au lancement" >> $B; exit 1; fi
R=$(basename $(grep -m1 -o "logs/rsl_rl/rhps1_tracking/[0-9_-]*" $S/rhps1_runs/st21.log)); echo $R > $S/rhps1_runs/st21.run
echo "st21 lance PID $P -> $R $(date)" >> $B
D=logs/rsl_rl/rhps1_tracking/$R
for it in $(seq 13750 250 17497); do
  until [ -f $D/model_$it.pt ] || ! [ -d /proc/$P ]; do sleep 30; done
  [ -f $D/model_$it.pt ] || it=$(ls $D | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1 | tr -dc 0-9)
  sleep 20
  (source $S/stairs/env.sh; export RHPS1_STAIRS_H=0.005,0.18 RHPS1_STAIRS_SPEED=0.085 RHPS1_STAIRS_SCAN_NOISE=0 RHPS1_STAIRS_VEL_TAU=1.0 RHPS1_STAIRS_W_FOOTHOLD=-5 RHPS1_STAIRS_FOOTHOLD_PRE=1; BENCH_T=24 BENCH_PER=2 BENCH_FLAT=8 nice -n 5 timeout 3000 .venv/bin/python $S/stairs/stairs_bench.py $R model_$it.pt ST21_$it 2>/dev/null | sed -n '/^STAIRS/,$p') >> $B
  [ -d /proc/$P ] || break
done
echo "chain_st21 fini $(date)" >> $B
