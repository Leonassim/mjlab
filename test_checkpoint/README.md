# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique. Le but est de
pouvoir le relire sur une autre machine sans traîner des dizaines de fichiers
de 7 Mo dans l'historique git.

`2026-07-26_20-47-53_model_450.pt` — run 2026-07-26_20-47-53, itération 450/15000.

Premier run après la refonte des rewards (36 -> 31 termes, 6 -> 2 curricula).
À cette itération : trackv 2.90, fell_down 0.030, contacts à plat 2.81/4
(au-dessus du record précédent de 2.78), vitesse d'impact 0.088 sous la
limite de 0.15. Le défaut connu est la posture à l'arrêt : commande nulle,
le robot garde un pied levé ~27 mm en permanence (ARRET 0.96).

## Relecture

```
uv run play Mjlab-Velocity-Flat-RHPS1 --checkpoint-file test_checkpoint/2026-07-26_20-47-53_model_450.pt
```
