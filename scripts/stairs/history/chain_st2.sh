#!/usr/bin/env bash
# chain_st2.sh: after st1, the same stairs run on seed 7 (ST2 benches).
cd /home/lmoussafir/mjlab-rhps1
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
until [ -f $S/rhps1_runs/st1.tpid ]; do sleep 30; done
P0=$(cat $S/rhps1_runs/st1.tpid); while [ -d /proc/$P0 ]; do sleep 30; done; sleep 20
P=$(SEED=7 NUM_ENVS=${NUM_ENVS:-3584} $S/stairs/st.sh st2 4000 RHPS1_STAIRS_NJMAX=${NJMAX:-0} RHPS1_STAIRS_NCONMAX=${NCONMAX:-0})
echo $P > $S/rhps1_runs/st2.tpid
until grep -q -E "Learning iteration|Traceback|Error" $S/rhps1_runs/st2.log; do sleep 5; done
if grep -q -E "Traceback|OutOfMemory" $S/rhps1_runs/st2.log; then echo "st2 ECHEC au lancement" >> $B; exit 1; fi
R=$(basename $(grep -m1 -o "logs/rsl_rl/rhps1_tracking/[0-9_-]*" $S/rhps1_runs/st2.log)); echo $R > $S/rhps1_runs/st2.run
echo "st2 lance PID $P -> $R $(date)" >> $B
D=logs/rsl_rl/rhps1_tracking/$R
for it in $(seq 13750 250 17497); do
  until [ -f $D/model_$it.pt ] || ! [ -d /proc/$P ]; do sleep 30; done
  [ -f $D/model_$it.pt ] || it=$(ls $D | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1 | tr -dc 0-9)
  sleep 20
  (source $S/stairs/env.sh; BENCH_PER=2 BENCH_FLAT=8 nice -n 5 timeout 3000 .venv/bin/python $S/stairs/stairs_bench.py $R model_$it.pt ST2_$it 2>/dev/null | sed -n '/^STAIRS/,$p') >> $B
  [ -d /proc/$P ] || break
done
echo "chain_st2 fini $(date)" >> $B
