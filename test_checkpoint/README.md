# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique. Le but est de
pouvoir le relire sur une autre machine sans traîner des dizaines de fichiers
de 7 Mo dans l'historique git.

`2026-07-29_01-13-36_model_9150.pt` — run 2026-07-29_01-13-36, itération
9150/15000 (en cours). Le `.onnx` du même point est joint pour tester
directement sous mc_rtc sans réexporter.

## ATTENTION — code à utiliser avec ce checkpoint

Ce checkpoint a été entraîné avec le code **tel qu'il est commité ici**, en
particulier `_LEG_SCALE_MULTIPLIER = 7.0`.

Un paquet de changements est en préparation sur la machine d'entraînement et
**n'est pas poussé** : scale à 1.0, projection de faisabilité en couple dans
l'actionneur, `init_std`/`std_range`/`entropy_coef` recalés. Ils sont
incompatibles avec ce checkpoint — au scale 1.0 la même sortie réseau commande
1/7 de l'amplitude articulaire. Si un `uv run play` donne un robot mou ou
immobile, c'est ce décalage-là, pas la politique.

## Où en est ce run

Mesuré autour de l'itération 9180 (moyenne sur les 80 derniers points) :

| indicateur | valeur | note |
|---|---|---|
| contacts à plat | 2.84 / 4 | meilleur point du run, progression continue depuis 2.67 |
| vitesse d'impact | 0.064 | tenu depuis l'itération 5500 |
| jerk d'action | 0.177 | stable |
| suivi de vitesse | 2.71 | stable |
| chutes par épisode | 0.100 | descendu de 0.13 |
| appui simple à l'arrêt | 0.071 | continue de baisser |
| balancement corps (lin/ang) | 0.068 / 0.235 | toujours plat depuis l'introduction du terme |
| hauteur de pas | 0.0060 | **toujours bloqué**, cible 0.08 |
| glissement | 0.0193 | |
| ratio demande PD / limite couple | 3.27 | remonté depuis 2.74 |
| ratio couple appliqué / limite | 0.745 | le couple appliqué est aux trois quarts de la limite en moyenne |
| std d'exploration | 1.00 | 0.43 → 0.98 sur les 1000 premières itérations, puis plat 8000 itérations |
| récompense moyenne | 10.02 | budget : +11.1 de tâche contre -10.3 de pénalités, ratio 0.93:1 |

## Déploiement

**Résolu depuis le checkpoint précédent** : le mode torque-clamp sous
mc_mujoco fonctionne. La cause du "ça ne marche pas du tout" était un bug du
contrôleur C++, pas la politique — mjlab incrémente
`_elapsed_since_target_update` dans un hook `update(dt)` séparé appelé à chaque
sous-pas physique, alors que le C++ ne l'incrémentait que dans la branche
"cible inchangée", qui ne s'exécutait jamais. La vitesse estimée saturait donc
à sa limite à chaque pas et le terme kd demandait des couples en bang-bang.

**Pas encore validé** : le mode position/QP. Sans clamp de couple en aval du
PD dans mc_rtc, une politique dont la demande PD est à 3.27x la limite ne
transfère pas. C'est ce que le paquet non poussé corrige, et il faudra un
nouvel entraînement pour en profiter.

## Deux points ouverts sur ce run

- `peak_height_mean` à 0.006 pour une cible à 0.08, plat depuis des milliers
  d'itérations malgré `min_foot_height` à -100. Le poids n'est pas le
  problème.
- la std d'exploration s'équilibre à 1.00 (soit 0.049 rad, ~2.8°) et y reste,
  contre une cible de conception de 0.021 rad. Ce n'est pas un emballement
  (le plafond est à 1.3, elle ne le touche pas) mais un équilibre fixé par
  `entropy_coef`.

## Relecture

```
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-07-29_01-13-36_model_9150.pt \
  --num-envs 1 --fast True
```

Les booléens de tyro exigent une valeur explicite : `--fast True`, pas
`--fast` seul. `--print-impact-vel True` affiche vitesse d'impact et pic de
hauteur à chaque pose.
