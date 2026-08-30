#!/usr/bin/env bash
# Ecart a la cible BWC, toutes les 20 min. L'objectif n'est pas "converger",
# c'est de rejoindre un comportement chiffre.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
for round in $(seq 1 "${2:-60}"); do
  sleep "${1:-1200}"
  RUN=$(ls -1dt logs/rsl_rl/rhps1_velocity/*/ | head -1)
  echo "===== $round  $(date +%H:%M) ====="
  .venv/bin/python - "$RUN" <<'PY'
from tensorboard.backend.event_processing import event_accumulator as ea
import glob, sys
a=ea.EventAccumulator(sorted(glob.glob(sys.argv[1]+"events.out*"))[-1],size_guidance={ea.SCALARS:0}); a.Reload()
T=a.Tags()["scalars"]
def last(t,n=200):
    if t not in T: return float('nan')
    v=[e.value for e in a.Scalars(t)][-n:]
    return sum(v)/len(v) if v else float('nan')
it=a.Scalars("Episode_Reward/track_linear_velocity")[-1].step
print(f"{sys.argv[1].split('/')[-2]}  it {it}")
rows=[("clearance","Metrics/sole_clearance_p90",0.063),
      ("capture_x","Metrics/capture_err_x",0.015),
      ("offset CoM","Metrics/com_stance_offset",0.070),
      ("periode","Metrics/step_period_mean",0.900),
      ("double app","Metrics/double_support_frac",None),
      ("impact","Metrics/landing_vel_mean",None),
      ("couple","Metrics/torque_limit_ratio_mean",None),
      ("chutes","Episode_Termination/fell_down",None),
      ("track","Episode_Reward/track_linear_velocity",None)]
print(f"{'':<12}{'actuel':>9}{'BWC':>9}{'% cible':>9}")
for nm,tag,tgt in rows:
    v=last(tag)
    if tgt: print(f"{nm:<12}{v:9.4f}{tgt:9.4f}{100*v/tgt:8.0f}%")
    else:   print(f"{nm:<12}{v:9.4f}{'--':>9}")
PY
done
