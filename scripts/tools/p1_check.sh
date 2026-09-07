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
# Le couple se juge contre la POLICY 0. Les limites du MODELE sont les bonnes --
# ce sont des limites operationnelles, pas des maxima moteur, et l'ecretage
# existe pour empecher le robot de faire n'importe quoi. Detour du 2026-09-04 :
# j'etais alle chercher les pics de la fiche technique (genou 180.8 contre 100)
# en croyant que mes mesures surestimaient la gravite. Faux : ce que le moteur
# PEUT fournir n'est pas ce qu'on AUTORISE.
#
# La policy 0 est la seule politique dont on SAIT qu'elle est acceptable sur le
# robot, donc elle dit quel niveau d'ecretage passe. En mesure deterministe :
#
#   saturation jambes (banc)              0.0103
#   demande avant ecretage (en marche)    total exces^2 0.1203, ratio max 4.91
#
# La saturation d'ENTRAINEMENT affichee ici n'est pas comparable a ces chiffres
# (action echantillonnee, randomisation) : elle sert de signal de tendance, et
# le verdict se prend en deterministe au banc contre les valeurs ci-dessus.
# lift : cible 0.045 et non 0.060. Les "7.1 cm du BWC" que je citais etaient son
# MAXIMUM sur 20 cycles ; sa mediane reelle est 3.2 cm. La fourchette de Leo,
# "3 a 5 cm minimum", est la bonne reference, et l'index 11 a 4.4 cm y etait deja.
#
# demiPer : Metrics/step_period_mean compte l'intervalle entre deux poses de
# N'IMPORTE QUEL pied (landed = first.any(dim=1)), donc un DEMI-cycle. La cible
# 0.45 correspond au cycle complet de 0.887 s du BWC. Lu comme un cycle complet,
# il ferait croire a un facteur deux qui n'existe pas.
W=[("reward","Train/mean_reward",None,None),
   ("impact","Metrics/landing_vel_mean","<=",0.100),
   ("falls","Episode_Termination/fell_down","<=",0.010),
   ("lift","Metrics/sole_height_p90",">=",0.045),
   # Marche sur la pointe. Repere par Leo a l'oeil sur les videos, confirme par
   # la mesure : le contact talon s'effondre le long de la lignee, 0.60 sur la
   # 6/6, 0.42 sur l'index 11, 0.23 sur P1b, pendant que l'inclinaison de
   # semelle a la pose est multipliee par quatre (0.024 -> 0.103 rad).
   # Aucun de mes trois criteres ne le voyait.
   # BILATERAL. contact_heel_frac est la PART DE CHARGE sur les boites arriere,
   # pas un temps de contact : 0.5 = centre, au-dessus = sur les talons, en
   # dessous = sur les pointes. Ma premiere version demandait ">= 0.50", c'est-a
   # -dire d'etre sur les TALONS -- exactement le defaut que Leo avait observe
   # sur le vrai robot, note dans rewards.py:3010. Le critere est l'ECART AU
   # CENTRE.
   #   6/6 0.603 (+0.103 talons)   index 11 0.425 (-0.075)
   #   P1b 0.228 (-0.272, le defaut vu en video)   P2c 0.392 (-0.108)
   ("|talon-.5|","Metrics/contact_heel_frac","centre",0.500),
   ("tiltPose","Metrics/sole_tilt_touchdown","<=",0.040),
   ("--","",None,None),
   ("refErr","Metrics/bwc_ref_err_rad",".",0.030),
   ("demiPer","Metrics/step_period_mean",".",0.450),
   ("satleg","Metrics/torque_saturated_frac_legs",".",0.200),
   ("refFrac","Metrics/bwc_ref_active_frac",None,None)]
show=its[::max(1,len(its)//6)][-6:]
print(f"{'':9s}"+"".join(f"{i:>10d}" for i in show)+"   cible")
for n,tag,op,tgt in W:
    if n=="--":
        print("  --- diagnostics (n'emportent pas le verdict) ---"); continue
    d=S(tag)
    if not d: continue
    f=(lambda x: abs(x-tgt)) if op=="centre" else (lambda x: x)
    row="".join(f"{f(d[i]):10.4f}" if i in d else f"{'--':>10s}" for i in show)
    c=("" if op is None else
       (f"   suivi ~{tgt}" if op=="." else
        ("   <= 0.10 (ecart au centre)" if op=="centre" else f"   {op} {tgt}")))
    print(f"{n:9s}{row}{c}")
# convergence : plateau du reward sur le dernier quart
r=[S("Train/mean_reward")[i] for i in its]
if len(r)>400:
    import statistics as st
    a,b=r[-400:-200],r[-200:]
    ga=(st.fmean(b)-st.fmean(a))/max(abs(st.fmean(a)),1e-6)
    print(f"\nplateau : reward {st.fmean(a):.1f} -> {st.fmean(b):.1f}  ({ga*100:+.1f} % sur 200 iterations)")
PY
