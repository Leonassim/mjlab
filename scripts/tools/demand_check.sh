#!/usr/bin/env bash
# Un tic de surveillance pour T2. Court, se termine tout seul, et affiche
# les criteres -- un moniteur qui journalise sans reveiller personne a deja
# coute une nuit d'entrainement a ce projet.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
R=$(ls -dt logs/rsl_rl/rhps1_velocity/*/ 2>/dev/null | head -1)
if ! pgrep -f "[t]rain Mjlab-Velocity-Flat-RHPS1" >/dev/null; then
  echo "ARRET: plus de train en cours"
fi
.venv/bin/python - "$R" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator(sys.argv[1],size_guidance={'scalars':0}); ea.Reload()
tags=ea.Tags()['scalars']
def series(t):
  return [(e.step,e.value) for e in ea.Scalars(t)] if t in tags else []
its=[s for s,_ in series("Train/mean_reward")]
if not its:
  print("aucune iteration encore"); sys.exit()
print(f"run {sys.argv[1].rstrip('/').split('/')[-1]}   iteration {its[-1]}")
WATCH=[("reward","Train/mean_reward"),("falls","Episode_Termination/fell_down"),
       ("lift","Metrics/sole_height_p90"),("impact","Metrics/landing_vel_mean"),
       ("period","Metrics/step_period_mean"),("satleg","Metrics/torque_saturated_frac_legs"),
       ("dmdMax","Metrics/torque_demand_ratio_max"),("dmdMean","Metrics/torque_demand_ratio_mean"),
       ("dmdFrac","Metrics/torque_demand_over_frac"),("Rdmd","Episode_Reward/torque_demand")]
show=its[::max(1,len(its)//6)][-6:]
print(f"{'':10s}"+"".join(f"{i:>10d}" for i in show))
for n,t in WATCH:
  d=dict(series(t))
  if not d: continue
  print(f"{n:10s}"+"".join(f"{d[i]:10.4f}" if i in d else f"{'--':>10s}" for i in show))
PY
