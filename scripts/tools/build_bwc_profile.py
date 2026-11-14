"""Construire le profil articulaire de reference depuis un log BaselineWalking.

Le profil dit, pour chaque articulation et chaque phase du cycle, l'angle que le
BWC tient. C'est une reference FAISABLE : elle tourne sur le vrai robot, donc la
suivre respecte les limites de couple par construction -- contrairement a un
barreme de recompenses dont chaque seuil doit etre devine.

ALIGNEMENT SUR LE SOMMET DU VOL, pas mediane point par point. Le sommet du vol ne
tombe pas a la meme phase normalisee d'un cycle a l'autre (il s'etale sur 56 % du
cycle), donc une mediane brute aplatit le pic : 2.4 cm de lever au lieu de 3.2
reels, et 7.4 deg de hanche au lieu de 19.5. Mesure, pas suppose.

  uv run python scripts/tools/build_bwc_profile.py <log.bin> [sortie.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

# Ordre du module mc_rtc RHPS1 en mode mujoco (rhps1.cpp), jambes d'abord.
MODULE_ORDER = [
  "L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P",
  "CHEST_Y", "CHEST_P",
  "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P",
]
N_BINS = 50
DT = 0.005


def main() -> None:
  src = Path(sys.argv[1])
  out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/bwc_gait_profile.json")
  import mc_log_ui

  log = mc_log_ui.read_log(str(src))
  t = np.asarray(log["t"])
  q = np.column_stack([log[f"qIn_{i}"] for i in range(len(MODULE_ORDER))])
  fL = np.asarray(log["LeftFootForceSensor_fz"])
  fR = np.asarray(log["RightFootForceSensor_fz"])
  zL = np.asarray(log["FootTask_Left_pose_tz"])

  # Fenetre ou le robot marche vraiment.
  vw = np.column_stack([log[f"FloatingBase_linearVelocity_{a}"] for a in "xyz"])
  wz = np.asarray(log["FloatingBase_angularVelocity_z"])
  mv = (np.abs(vw[:, 0]) + np.abs(vw[:, 1]) + np.abs(wz)) > 0.03
  i0, i1 = np.where(mv)[0][[0, -1]]

  # Cycles delimites par les poses du pied GAUCHE : phase 0 = touchdown gauche.
  #
  # HYSTERESIS 20/250 N, pas un seuil unique. A 60 N la charge rebondit pendant
  # l'appui et coupe un cycle en deux : 35 "poses" pour une periode mediane de
  # 1.02 s, alors que l'intervalle reel entre deux poses du meme pied est de
  # l'ordre de 1.6-1.8 s. Le profil construit dessus avait son sommet de vol a
  # 46 % du cycle apres la pose, la ou il doit tomber vers 75 % -- le pied gauche
  # est en appui la premiere moitie du cycle qui commence par sa propre pose.
  st, cL = False, np.zeros(len(fL), bool)
  for k in range(i0, i1):
    if not st and fL[k] > 250.0:
      st = True
    elif st and fL[k] < 20.0:
      st = False
    cL[k] = st
  td = np.where(np.diff(cL.astype(int)) == 1)[0]
  if len(td) < 6:
    raise SystemExit(f"seulement {len(td)} poses du pied gauche : log trop court")

  # Le log est une conduite manuelle : arrets, demi-tours, pas de cadence unique.
  # On ne garde que les cycles proches du mode, sinon le reechantillonnage a 50
  # points melange des cycles de durees incomparables.
  dur = np.diff(td) * DT
  mode = float(np.median(dur[(dur > 0.8) & (dur < 3.0)]))

  z0 = np.percentile(zL[i0:i1], 5)
  cycles, lifts = [], []
  for a, b in zip(td[:-1], td[1:]):
    n = b - a
    if not (0.75 < n * DT / mode < 1.25):    # +/-25 % autour du mode
      continue
    u = np.linspace(0, 1, n, endpoint=False)
    g = np.linspace(0, 1, N_BINS, endpoint=False)
    cyc = np.column_stack([np.interp(g, u, q[a:b, j]) for j in range(q.shape[1])])
    lift = np.interp(g, u, zL[a:b] - z0)
    cycles.append(cyc)
    lifts.append(lift)

  C = np.stack(cycles)                        # [n_cycles, N_BINS, n_joints]
  L = np.stack(lifts)
  # Aligner chaque cycle sur SON sommet de vol, puis mediane.
  peak = int(np.median(np.argmax(L, axis=1)))
  for k in range(len(C)):
    C[k] = np.roll(C[k], peak - int(np.argmax(L[k])), axis=0)
    L[k] = np.roll(L[k], peak - int(np.argmax(L[k])))
  prof = np.median(C, axis=0)
  lift_med = np.median(L, axis=0)

  # POSTURE D'ARRET. Le log contient les deux regimes : 32.7 s a l'arret sur 90.
  # Mesuree, elle n'est qu'a 2-3 deg de la posture moyenne de marche (genou 26.4
  # contre 29.2) -- la transition arret/marche est une amplitude qui s'ouvre
  # autour d'une posture quasi identique, pas un changement de configuration.
  # C'est ce qui permet de traiter les deux regimes avec UNE reference et une
  # interpolation, au lieu de deux comportements a apprendre separement.
  spd = np.abs(vw[:, 0]) + np.abs(vw[:, 1]) + np.abs(wz)
  still = (spd < 0.02) & (fL > 100.0) & (fR > 100.0)
  still[:i0] = False
  if still.sum() < 200:
    raise SystemExit("pas assez d'arret dans le log pour une posture de repos")
  stand = np.median(q[still], axis=0)

  period = float(np.median(np.diff(td)) * DT)
  data = {
    "source": src.name,
    "method": ("cycles delimites par les poses du pied gauche, reechantillonnes a "
               f"{N_BINS} points puis ALIGNES sur le sommet du vol avant mediane. "
               "Une mediane point par point brute aplatit le pic : le sommet ne "
               "tombe pas a la meme phase normalisee d'un cycle a l'autre."),
    "n_cycles": int(len(C)),
    "period_s": period,
    "foot_lift_median_m": float(lift_med.max()),
    "foot_lift_max_m": float(L.max()),
    "phase_bins": N_BINS,
    "peak_bin": peak,
    "joints": {n: [float(x) for x in prof[:, j]] for j, n in enumerate(MODULE_ORDER)},
    "standing_s": float(still.sum() * DT),
    "standing": {n: float(stand[j]) for j, n in enumerate(MODULE_ORDER)},
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(data, indent=1))
  print(f"ecrit {out}")
  print(f"  {len(C)} cycles   periode {period:.3f} s   lever median "
        f"{lift_med.max()*100:.2f} cm   max {L.max()*100:.2f} cm")
  print(f"\n  {'joint':14}{'amplitude du profil':>20}")
  for j, n in enumerate(MODULE_ORDER):
    if "CHEST" in n:
      continue
    amp = np.degrees(prof[:, j].max() - prof[:, j].min())
    print(f"  {n:14}{amp:17.2f} deg")


if __name__ == "__main__":
  main()
