# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique. Le but est de
pouvoir le relire sur une autre machine sans traîner des dizaines de fichiers
de 7 Mo dans l'historique git.

`2026-07-26_20-47-53_model_450.pt` — run 2026-07-26_20-47-53, itération 450/15000.

Premier run après la refonte des rewards (36 -> 31 termes, 6 -> 2 curricula).
Encore très tôt dans l'entraînement : c'est un début de marche, pas un
résultat.

## Où en est ce run

Mesuré autour de l'itération 550 :

| indicateur | valeur | note |
|---|---|---|
| contacts à plat | 2.91 / 4 | au-dessus du record précédent de 2.78 |
| garde au sol du pied | 15.3 mm | franchit la cible de 15 mm |
| vitesse d'impact | 0.085 | limite à 0.15 |
| suivi de vitesse | 2.71 | |
| chutes par épisode | 0.17 | |
| appui simple à l'arrêt | 0.974 | **défaut connu, non résolu** |

Les deux premières lignes sont les métriques que la refonte visait, et elles
dépassent toutes les valeurs jamais atteintes sur ce robot. La dernière est le
défaut ouvert : à commande nulle le robot tient une posture de cigogne, un pied
levé ~27 mm en permanence, toujours le même. Quatre hypothèses ont été testées
et réfutées (mesure fausse, transition inévitable, bruit d'exploration,
gradient manquant) ; la cause reste inconnue.

## Relecture

```
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-07-26_20-47-53_model_450.pt \
  --num-envs 1 --fast True
```

Les booléens de tyro exigent une valeur explicite : `--fast True`, pas
`--fast` seul. `--print-impact-vel True` affiche vitesse d'impact et pic de
hauteur à chaque pose, c'est-à-dire les deux mesures qui étaient fausses avant
cette refonte.
