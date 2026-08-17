# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique.

`2026-08-17_00-34-31_model_5250.pt` — run **2026-08-17_00-34-31, toujours en
cours**, itération 5250/15000. `.onnx` du même point joint. `env.yaml` est la
config effective de ce run.

## Ce que ce run change

Retour à l'objectif ET à la randomisation de la **policy 0** (run
2026-07-10_20-59-17, la seule config ayant marché sur robot réel), en gardant
les fonctions de récompense corrigées depuis :

- **Randomisation strictement celle de la policy 0** : 5 événements seulement
  (`foot_friction` partagée au démarrage 0.5–0.9, `encoder_bias` ±0.015,
  `base_com`, `reset_base`, `reset_robot_joints` sans dispersion). Vérifié par
  diff exhaustif contre son `env.yaml` : zéro écart sur la randomisation.
- **Noyaux de suivi de vitesse** remis à 0.20 / 0.35 (policy 0). À 0.40 un robot
  immobile encaissait déjà 78 % du terme contre 43 % à 0.20.
- **Termes ajoutés depuis, retirés** : `action_jerk` −45, `standing_pose` −40,
  `standing_joint_vel`, `standing_base_motion`. Aucun n'a jamais été isolé sur
  matériel.
- **Conservés délibérément** : le filtre de PostureTask (K=1600, modélise les
  ~25 ms de retard du QP), `effort_limit` 70 N·m au genou avec son échelle
  d'action correspondante (sécurité matériel), et les six termes de proximité.

Les poids élevés qui restent (`foot_slip` −28, `flat_support` −11,
`foot_clearance` −35) ne sont **pas** un objectif plus dur : ce sont des
recalibrages sur des fonctions corrigées qui rendent des valeurs des ordres de
grandeur plus petites. Six termes de la policy 0 ne mesuraient pas ce qu'ils
annonçaient (voir commit 12b25ac9) — elle a marché malgré eux, pas grâce à eux.

## Métriques d'entraînement, très au-dessus des runs précédents

Fenêtre it ≥ 4800 :

| | run précédent (v5) | **ce run** |
|---|---|---|
| `progress_ratio` | 0.62 | **0.85** (dernier 0.90) |
| `fell_down` | 0.34 | **0.07** |
| `sole_height_p90` | 0.029 | 0.024 |
| `air_time_mean` | 0.264 | 0.231 |

Le robot suit sa commande à 85 % en tombant cinq fois moins. Confirmé
visuellement sur les vidéos d'entraînement.

## Ce qu'il faut vérifier, et pourquoi ce test compte

Tous les runs précédents avaient de bonnes métriques d'entraînement et un robot
**immobile en `play`** — un écart jamais expliqué malgré une revue complète du
code (config, boucle, normalisation, filtre, commandes : tout identique).

Ce run est le premier avec la randomisation **stricte de la policy 0**. Donc :

- si le robot marche en `play` → l'écart venait de la randomisation, et on a la
  réponse ;
- s'il reste immobile → la randomisation est définitivement écartée et il faut
  chercher ailleurs.

Dans les deux cas c'est informatif. **Juger visuellement, pas avec un script** :
les bancs de mesure externes ont lu 6 à 20× trop bas toute la semaine.

## Reproduire

```bash
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-08-17_00-34-31_model_5250.pt \
  --gamepad true
```

`RHPS1_PLAY_CMD_SCALE=1.3` en préfixe pousse la commande max à 0.39 m/s (hors
distribution d'entraînement, uniquement pour sonder la réactivité).

Le run continue en arrière-plan ; un checkpoint plus tardif sera probablement
meilleur.
