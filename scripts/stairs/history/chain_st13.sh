#!/usr/bin/env bash
# chain_st13.sh: st12 + reference-free double support 1 (slip terms alone push the gait to 86 % single support).
cd /home/lmoussafir/mjlab-rhps1
S=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
B=$S/stairs/summary.txt
P=$(NUM_ENVS=${NUM_ENVS:-3584} $S/stairs/st.sh st13 4000 RHPS1_STAIRS_NJMAX=${NJMAX:-0} RHPS1_STAIRS_NCONMAX=${NCONMAX:-0} RHPS1_STAIRS_H=0.005,0.18 RHPS1_STAIRS_INIT_LEVEL=0 RHPS1_STAIRS_EPISODE_S=16 RHPS1_STAIRS_W_TEACHER=0 RHPS1_STAIRS_W_ARRIVAL=0 RHPS1_STAIRS_W_CLEAR=2 RHPS1_STAIRS_W_STATIC=0 RHPS1_STAIRS_SPEED=0.085 RHPS1_STAIRS_SCAN_NOISE=0 RHPS1_STAIRS_W_ALIVE=8 RHPS1_STAIRS_W_DS=1)
echo $P > $S/rhps1_runs/st13.tpid
until grep -q -E "Learning iteration|Traceback|Error" $S/rhps1_runs/st13.log; do sleep 5; done
if grep -q -E "Traceback|OutOfMemory" $S/rhps1_runs/st13.log; then echo "st13 ECHEC au lancement" >> $B; exit 1; fi
R=$(basename $(grep -m1 -o "logs/rsl_rl/rhps1_tracking/[0-9_-]*" $S/rhps1_runs/st13.log)); echo $R > $S/rhps1_runs/st13.run
echo "st13 lance PID $P -> $R $(date)" >> $B
D=logs/rsl_rl/rhps1_tracking/$R
for it in $(seq 13750 250 17497); do
  until [ -f $D/model_$it.pt ] || ! [ -d /proc/$P ]; do sleep 30; done
  [ -f $D/model_$it.pt ] || it=$(ls $D | grep -E '^model_[0-9]+\.pt$' | sort -t_ -k2 -n | tail -1 | tr -dc 0-9)
  sleep 20
  (source $S/stairs/env.sh; export RHPS1_STAIRS_H=0.005,0.18 RHPS1_STAIRS_SPEED=0.085 RHPS1_STAIRS_SCAN_NOISE=0; BENCH_PER=2 BENCH_FLAT=8 nice -n 5 timeout 3000 .venv/bin/python $S/stairs/stairs_bench.py $R model_$it.pt ST13_$it 2>/dev/null | sed -n '/^STAIRS/,$p') >> $B
  [ -d /proc/$P ] || break
done
echo "chain_st13 fini $(date)" >> $B
