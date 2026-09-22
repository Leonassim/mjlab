"""Extraire d'un log BaselineWalking le CSV attendu par csv_to_npz.py.

Format de csv_to_npz.MotionLoader._load_motion :
    colonnes 0-2   position du flottant (monde)
    colonnes 3-6   quaternion du flottant, XYZW (le script le reordonne en WXYZ)
    colonnes 7+    angles articulaires, dans l'ordre du modele MuJoCo

L'ordre des colonnes articulaires est celui du MODELE mjlab, pas celui du module
mc_rtc : les deux listes contiennent les memes 30 noms dans un ordre
completement different (joint_names commence par CHEST_Y, le module par
L_CROTCH_Y). Confondre les deux a deja produit un tableau de couples entierement
mal etiquete.

  uv run python scripts/tools/bwc_log_to_csv.py <log.bin> <sortie.csv> [--fps 200]
"""

import argparse
from pathlib import Path

import numpy as np

# Ordre du module mc_rtc RHPS1 en mode mujoco (mc_rhps1/src/rhps1.cpp), qui est
# l'ordre des colonnes qIn_* du log.
MODULE_ORDER = [
  "L_CROTCH_Y", "L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "L_ANKLE_P",
  "CHEST_Y", "CHEST_P",
  "R_CROTCH_Y", "R_CROTCH_R", "R_CROTCH_P", "R_KNEE_P", "R_ANKLE_R", "R_ANKLE_P",
  "HEAD_Y", "HEAD_P",
  "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P", "L_ELBOW_Y",
  "L_WRIST_R", "L_WRIST_Y",
  "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", "R_ELBOW_Y",
  "R_WRIST_R", "R_WRIST_Y",
]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("log")
  p.add_argument("out")
  p.add_argument("--model-order", required=True,
                 help="fichier contenant l'ordre articulaire du modele mjlab")
  p.add_argument("--min-speed", type=float, default=0.03,
                 help="ne garder que les instants ou le robot bouge")
  p.add_argument("--keep-still", action="store_true",
                 help="garder aussi les phases d'arret (transitions incluses)")
  a = p.parse_args()

  # mc_log_ui tire PyQt5, absent de l'environnement uv de mjlab : ce script
  # tourne donc sous le python SYSTEME. L'ordre articulaire du modele lui est
  # passe par --model-order (une ligne, noms separes par des virgules), produit
  # cote mjlab. Les deux listes ont les memes 30 noms dans un ordre different,
  # et les confondre a deja produit un tableau entierement mal etiquete.
  import mc_log_ui

  log = mc_log_ui.read_log(a.log)
  n_log = len(MODULE_ORDER)
  q_mod = np.column_stack([log[f"qIn_{i}"] for i in range(n_log)])
  pos = np.column_stack([log[f"FloatingBase_position_{c}"] for c in "xyz"])
  quat = np.column_stack([log[f"FloatingBase_orientation_{c}"] for c in "xyzw"])
  # mc_rtc logs the INVERSE rotation (SpaceVecAlg convention). Conjugate it:
  # the raw quaternion made stance feet slide at p90 179 mm/s instead of 2.5.
  quat[:, :3] *= -1.0

  vw = np.column_stack([log[f"FloatingBase_linearVelocity_{c}"] for c in "xyz"])
  wz = np.asarray(log["FloatingBase_angularVelocity_z"])
  speed = np.abs(vw[:, 0]) + np.abs(vw[:, 1]) + np.abs(wz)
  moving = speed > a.min_speed
  idx = np.where(moving)[0]
  lo, hi = idx[0], idx[-1] + 1
  keep = slice(lo, hi) if a.keep_still else None
  sel = np.arange(lo, hi) if a.keep_still else idx[(idx >= lo) & (idx < hi)]
  del keep

  model_order = [x for x in Path(a.model_order).read_text().strip().split(",") if x]
  perm = [MODULE_ORDER.index(n) for n in model_order]
  assert len(perm) == n_log, f"{len(perm)} colonnes appariees sur {n_log}"

  out = np.hstack([pos[sel], quat[sel], q_mod[sel][:, perm]])
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(a.out, out, delimiter=",", fmt="%.8f")
  print(f"ecrit {a.out}  {out.shape[0]} images  {out.shape[1]} colonnes")
  print(f"  fenetre {lo * 0.005:.1f} - {hi * 0.005:.1f} s   "
        f"{'arrets conserves' if a.keep_still else 'mouvement seul'}")
  print(f"  ordre articulaire du modele : {model_order[:4]} ...")


if __name__ == "__main__":
  main()
