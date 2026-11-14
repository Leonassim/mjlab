#!/usr/bin/env bash
# UN controle puis on rend la main, pour que le harnais me reveille. Une boucle
# de 20 h n'emet aucune notification : elle journalise, elle n'alerte pas.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
sleep "${1:-1200}"
RUN=$(ls -1dt logs/rsl_rl/rhps1_velocity/*/ | head -1)
.venv/bin/python - "$RUN" <<'PY'
from tensorboard.backend.event_processing import event_accumulator as ea
import glob, sys
a=ea.EventAccumulator(sorted(glob.glob(sys.argv[1]+"events.out*"))[-1],size_guidance={ea.SCALARS:0}); a.Reload()
T=a.Tags()["scalars"]
def ser(t): return [(e.step,e.value) for e in a.Scalars(t)] if t in T else []
def mean(t,n=150):
    v=[x for _,x in ser(t)][-n:]; return sum(v)/len(v) if v else float('nan')
def slope(t,n=600):
    d=ser(t)[-n:]
    if len(d)<20: return float('nan')
    h=len(d)//2
    return sum(x for _,x in d[h:])/len(d[h:]) - sum(x for _,x in d[:h])/len(d[:h])
it=ser("Episode_Reward/track_linear_velocity")[-1][0]
print(f"{sys.argv[1].split('/')[-2]}  it {it}")
rows=[("steplen","Metrics/step_length_mean",0.05),("air","Metrics/air_time_mean",0.765),
      ("periode","Metrics/step_period_mean",0.900),("clearance","Metrics/sole_clearance_p90",0.063),
      ("offset","Metrics/com_stance_offset",0.070),("impact","Metrics/landing_vel_mean",None),
      ("couple","Metrics/torque_limit_ratio_mean",None),("chutes","Episode_Termination/fell_down",None),
      ("track","Episode_Reward/track_linear_velocity",None),
      ("sat jambes","Metrics/torque_saturated_frac_legs",0.25),
      ("sat haut","Metrics/torque_saturated_frac_upper",0.25)]
print(f"{'':<10}{'actuel':>9}{'pente':>9}{'BWC/seuil':>11}")
for nm,tag,tgt in rows:
    print(f"{nm:<10}{mean(tag):9.4f}{slope(tag):+9.4f}" + (f"{tgt:11.4f}" if tgt else f"{'--':>11}"))
# faisabilite : c'est l'articulation la plus saturee qui decide, pas la moyenne
sat=[(mean(t),t.split('/')[1]) for t in T if t.startswith("TorqueSat/")]
if sat:
    sat.sort(reverse=True)
    print("  pire articulation : " + ", ".join(f"{n} {v:.3f}" for v,n in sat[:3]))
PY
