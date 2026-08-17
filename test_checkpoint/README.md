# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique.

`2026-08-17_00-34-31_model_10350.pt` — run **2026-08-17_00-34-31, toujours en
cours**, itération 10350/15000. `.onnx` du même point joint. `env.yaml` est la
config effective de ce run.

## Reproduire

```bash
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-08-17_00-34-31_model_10350.pt \
  --gamepad true
```

`RHPS1_PLAY_CMD_SCALE=1.3` en préfixe pousse la commande max à 0.39 m/s (hors
distribution d'entraînement, uniquement pour sonder la réactivité).

## Ce que ce run change

Retour à l'objectif ET à la randomisation de la **policy 0** (run
2026-07-10_20-59-17, la seule config ayant marché sur robot réel), en gardant
les fonctions de récompense corrigées depuis :

- **Randomisation strictement celle de la policy 0** : 5 événements seulement
  (`foot_friction` partagée au démarrage 0.5–0.9, `encoder_bias` ±0.015,
  `base_com`, `reset_base`, `reset_robot_joints` sans dispersion).
- **Noyaux de suivi de vitesse** remis à 0.20 / 0.35 (policy 0).
- **Termes ajoutés depuis, retirés** : `action_jerk` −45, `standing_pose` −40,
  `standing_joint_vel`, `standing_base_motion`.
- **Conservés délibérément** : le filtre de PostureTask (K=1600), `effort_limit`
  70 N·m au genou avec son échelle d'action correspondante (sécurité matériel),
  et les six termes de proximité.

## Où on en est

**L'écart train/play est résolu.** C'était bien la randomisation : revenir à
celle de la policy 0 a suffi. Ce run marche en `play` comme dans les vidéos
d'entraînement, confirmé visuellement le 2026-08-17 (à partir de ~5250).

Les métriques du tableau ci-dessous datent de la fenêtre it ≥ 4800 et n'ont pas
été remesurées à 10350 :

| | run précédent (v5) | ce run (it ≥ 4800) |
|---|---|---|
| `progress_ratio` | 0.62 | **0.85** |
| `fell_down` | 0.34 | **0.07** |
| `sole_height_p90` | 0.029 | 0.024 |
| `air_time_mean` | 0.264 | 0.231 |

**Le problème ouvert est le déploiement, pas la policy.** En mc_mujoco le robot
est instable même debout, sous QP comme en bypass. Quatre divergences de plant
trouvées et corrigées côté déploiement le 2026-08-17 — aucune ne touche `play`,
qui tourne dans le plant d'entraînement :

- mc_mujoco n'écrêtait **pas** le couple, là où mjlab fait
  `clamp(tau, ±force_limit)` à chaque pas (`pd_actuator.py:68`) — et avec une
  demande brute de 3–4× la limite, cet écrêtage mord en permanence. Mesuré 72×
  la limite sur 26 joints sur 30 en déploiement.
- `R_SHOULDER_P` tournait à **kp=500 au lieu de 15000** : `PDgains_sim.dat`
  était décalé d'un cran, écrit pour un ordre contenant `L_HAND`.
- le filtre de PostureTask (K=1600) manquait **en amont** de la différence
  finie et de la projection, alors que l'entraînement le place là.
- intégrateur Euler contre `implicitfast`.

Reste identifié mais non implémenté : l'**amortisseur de vitesse**
(`velocity_damper_di: 0.4`), absent du chemin bypass.

**Juger visuellement, pas avec un script** : les bancs de mesure externes ont lu
6 à 20× trop bas toute la semaine.

Le run continue en arrière-plan ; un checkpoint plus tardif sera probablement
meilleur.
