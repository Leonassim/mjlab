#!/usr/bin/env bash
# Point de convergence horaire sur la run en cours.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
sleep "${1:-3600}"
RUN=$(ls -1dt logs/rsl_rl/rhps1_velocity/*/ | head -1)
.venv/bin/python - "$RUN" <<'PY'
from tensorboard.backend.event_processing import event_accumulator as ea
import glob, sys
a=ea.EventAccumulator(sorted(glob.glob(sys.argv[1]+"events.out*"))[-1],size_guidance={ea.SCALARS:0}); a.Reload()
def s(t): return {e.step:e.value for e in a.Scalars(t)} if t in a.Tags()["scalars"] else {}
T={"track":"Episode_Reward/track_linear_velocity","clear":"Metrics/sole_clearance_p90",
   "torque":"Metrics/torque_limit_ratio_mean","falls":"Episode_Termination/fell_down",
   "period":"Metrics/step_period_mean","steplen":"Metrics/step_length_mean",
   "landing":"Metrics/landing_vel_mean","offset":"Metrics/com_stance_offset"}
S={k:s(v) for k,v in T.items()}
ks=sorted(S["track"].keys()); last=ks[-1]
print("RUN", sys.argv[1].split('/')[-2], "iteration", last)
def win(n,a,b):
    v=[S[n][k] for k in ks if a<=k<=b and k in S[n]]
    return sum(v)/len(v) if v else float('nan')
print(f"{'':<9}{'-2000':>10}{'-1000':>10}{'dernier1000':>12}{'derive%':>9}")
for n in T:
    x=win(n,last-2000,last-1000); y=win(n,last-1000,last)
    dr=100*(y-x)/abs(x) if x and x==x and abs(x)>1e-9 else float('nan')
    print(f"{n:<9}{x:10.4f}{y:10.4f}{y:12.4f}{dr:9.1f}")
PY
