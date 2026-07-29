# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique. Le but est de
pouvoir le relire sur une autre machine sans traîner des dizaines de fichiers
de 7 Mo dans l'historique git.

`2026-07-29_01-13-36_model_5550.pt` — run 2026-07-29_01-13-36, itération 5550/15000 (en cours).

Suite du run du 2026-07-27 (horloge de marche dépendante de la vitesse, cap
std corrigé). Quatre changements ciblés cette fois, tous jugés efficaces à ce
stade : `action_jerk` -45 → -90 (vibration), nouveau terme
`standing_base_motion` en norme L1 (balancement du corps entier à l'arrêt —
rien ne le pénalisait avant), coefficient `ANKLE_R` de `joint_torques_l2`
4.0 → 1.5 (`flat_support` était bloqué depuis des milliers d'itérations, ce
coefficient taxait le couple même qui achète un contact plat). Un cinquième
changement (réactivation de `torque_guidance_coef` pour la faisabilité des
couples) a provoqué un collapse net vers l'itération 1200 et a été annulé —
c'est la 5ᵉ reproduction documentée du même mode d'échec, voir `rl_cfg.py`.

## Où en est ce run

Mesuré autour de l'itération 5600 (moyenne sur ~80 points) :

| indicateur | valeur | note |
|---|---|---|
| contacts à plat | 2.82 / 4 | dépasse déjà le record précédent (2.91 atteint plus tard sur le run du 26/07, mais celui-ci était bloqué à 2.67 avant ce changement) |
| vitesse d'impact | 0.064 | meilleur que tous les runs précédents |
| jerk d'action | 0.174 | net progrès vs 0.24 avant `action_jerk` -90 |
| suivi de vitesse | 2.72 | |
| chutes par épisode | 0.13 | |
| appui simple à l'arrêt | 0.091 | bien meilleur que le 0.974 du run précédent |
| balancement corps (lin/ang) | 0.070 / 0.234 | nouvelle métrique, plate depuis son introduction — pas encore de baisse nette |
| ratio demande PD / limite couple | 2.74 | descendu depuis 5.6, remonte lentement depuis l'annulation du torque guidance |

**Non résolu à ce stade** : testé en direct sous mc_mujoco (mode normal, pas
torque), le retour utilisateur est que ça "ne marche pas du tout" —
contradictoire avec les métriques d'entraînement ci-dessus, cause non
identifiée. Sera creusé séparément ; les métriques de ce fichier reflètent
l'entraînement, pas encore une validation de déploiement.

## Relecture

```
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-07-29_01-13-36_model_5550.pt \
  --num-envs 1 --fast True
```

Les booléens de tyro exigent une valeur explicite : `--fast True`, pas
`--fast` seul. `--print-impact-vel True` affiche vitesse d'impact et pic de
hauteur à chaque pose.
