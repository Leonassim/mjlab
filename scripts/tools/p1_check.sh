#!/usr/bin/env bash
# Suivi de P1. Affiche les metriques qui portent les CIBLES MESUREES sur le BWC,
# pas celles que je surveillais avant et qui ne mesuraient pas la bonne chose.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
R=$(ls -dt logs/rsl_rl/rhps1_velocity/*/ 2>/dev/null | head -1)
pgrep -f "[t]rain Mjlab-Velocity-Flat-RHPS1" >/dev/null || echo "ARRET: plus de train"
.venv/bin/python - "$R" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator(sys.argv[1],size_guidance={'scalars':0}); ea.Reload()
t=ea.Tags()['scalars']
def S(tag): return {e.step:e.value for e in ea.Scalars(tag)} if tag in t else {}
its=sorted(S("Train/mean_reward"))
if not its: print("aucune iteration"); sys.exit()
print(f"run {sys.argv[1].rstrip('/').split('/')[-1]}   iteration {its[-1]}")
# CRITERES (portent le verdict) puis DIAGNOSTICS (expliquent, ne jugent pas).
# La distinction a ete perdue une fois : l'amplitude du genou en appui est ce
# qui EXPLIQUE le couple, pas un objectif -- en faire une porte reviendrait a
# exiger que la politique resolve le probleme comme le BWC.
#
# Le couple se juge contre la POLICY 0, pas contre un seuil invente ni contre la
# fiche technique. La fiche (91.71 N.m continu, 180.75 en pic 23.5 s) est trop
# permissive pour discriminer : l'index 11 tient 150 N.m par bouffees de 30 ms
# et reste dedans. La policy 0 est la seule politique dont on SAIT qu'elle
# marche sur le robot, et elle donne, en mesure deterministe :
#
#   saturation jambes (banc)              0.0103
#   demande avant ecretage (en marche)    total exces^2 0.1203, ratio max 4.91
#
# La saturation d'ENTRAINEMENT affichee ici n'est pas comparable a ces chiffres
# (action echantillonnee, randomisation) : elle sert de signal de tendance, et
# le verdict se prend en deterministe au banc contre les valeurs ci-dessus.
W=[("reward","Train/mean_reward",None,None),
   ("impact","Metrics/landing_vel_mean","<=",0.100),
   ("falls","Episode_Termination/fell_down","<=",0.010),
   ("lift","Metrics/sole_height_p90",">=",0.060),
   ("--","",None,None),
   ("refErr","Metrics/bwc_ref_err_rad",".",0.030),
   ("period","Metrics/step_period_mean",".",0.900),
   ("satleg","Metrics/torque_saturated_frac_legs",".",0.200),
   ("refFrac","Metrics/bwc_ref_active_frac",None,None)]
show=its[::max(1,len(its)//6)][-6:]
print(f"{'':9s}"+"".join(f"{i:>10d}" for i in show)+"   cible")
for n,tag,op,tgt in W:
    if n=="--":
        print("  --- diagnostics (n'emportent pas le verdict) ---"); continue
    d=S(tag)
    if not d: continue
    row="".join(f"{d[i]:10.4f}" if i in d else f"{'--':>10s}" for i in show)
    c="" if op is None else (f"   suivi ~{tgt}" if op=="." else f"   {op} {tgt}")
    print(f"{n:9s}{row}{c}")
# convergence : plateau du reward sur le dernier quart
r=[S("Train/mean_reward")[i] for i in its]
if len(r)>400:
    import statistics as st
    a,b=r[-400:-200],r[-200:]
    ga=(st.fmean(b)-st.fmean(a))/max(abs(st.fmean(a)),1e-6)
    print(f"\nplateau : reward {st.fmean(a):.1f} -> {st.fmean(b):.1f}  ({ga*100:+.1f} % sur 200 iterations)")
PY
