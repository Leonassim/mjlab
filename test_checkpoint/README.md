# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique.

`2026-08-16_10-52-02_model_4950.pt` — run **2026-08-16_10-52-02, toujours en
cours**, itération 4950/15000 (checkpoint intermédiaire, pas le run final).
`.onnx` du même point joint. `env.yaml` est la config effective de ce run.

## Ce que ce run change par rapport à 2026-08-12_20-36-28

Une seule différence, vérifiée par diff exhaustif contre l'`env.yaml` sauvegardé
du 12 août : `foot_friction.params.ranges` passe de `(0.4, 1.0)` à **`(0.5,
0.7)`** — même mode par-environnement/par-reset. Tout le reste (récompenses,
`std` de suivi de vitesse, curriculum, `effort_limit` genou 70 N·m, filtre de
PostureTask K=1600, limites de couple) est identique au bit près.

0,7 est la friction mesurée pour du caoutchouc dur sur parquet/lino ; ±0,3 mm
plutôt que jusqu'à 0,4 (très glissant), qui n'était jamais atteint par la
policy 0 (référence validée sur robot).

## Signal positif, mesuré dans la vraie boucle d'entraînement

`Metrics/progress_ratio` (ajouté ce jour) : vitesse d'avance projetée sur la
commande, `1.0` = suivi parfait, `0.0` = robot immobile. Mesuré à l'intérieur
de `runner.learn`, pas par un rollout externe — c'est la seule mesure jugée
fiable ce jour, voir plus bas.

| itération | `progress_ratio` | `sole_height_p90` | `fell_down` |
|---|---|---|---|
| ~1200 | 0,18 | 0,012 m | 0,26 |
| ~2800 | 0,58 | 0,022 m | 0,29 |
| ~5080 (dernier) | **0,57–0,60** | **0,025 m** | 0,32–0,40 |

Plateau stable depuis l'itération ~3000, aucune dérive. Pour comparaison, le
run du 12 août n'atteignait `sole_height_p90 = 0,020 m` qu'à l'itération
~4000 ; ce run l'atteint à ~2800.

## ATTENTION — non confirmé en dehors de la boucle d'entraînement

Tout test *externe* (n'importe quelle boucle `get_observations` → `policy` →
`step`, y compris `play`) donne un résultat très inférieur au `progress_ratio`
ci-dessus, et ce désaccord n'est pas expliqué :

- `scripts/tools/play_forward_check.py`, `play=True`, commande avant forcée
  0,2 m/s : déplacement quasi nul après 3 s (voir vidéo `/tmp/rhps1_play_forward.mp4`
  du 2026-08-16).
- 24 tirages indépendants sans seed (comme `play`), même commande : **0/24**
  au-dessus de 50 % du suivi, max 1,9 %. Écarte l'hypothèse du tirage
  malchanceux — ce n'est pas un effet de loterie de plant.
- Déterministe et stochastique donnent le même résultat (donc ce n'est pas le
  bruit d'exploration).
- La même signature avait déjà été mesurée le 2026-08-14 sur le run du 12 août :
  tous les bancs externes lisaient 6 à 20× en dessous de ce que `runner.learn`
  mesurait sur le même checkpoint.

Donc : **ne pas conclure "ça marche" avant test direct sur `play` ou
mc_mujoco.** Le signal d'entraînement est le meilleur de tous les runs testés
cette semaine, mais rien ne garantit qu'il se traduit en dehors de la boucle
d'entraînement — c'est précisément la question ouverte.

## Reproduire

```bash
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-08-16_10-52-02_model_4950.pt
```

Le run continue en arrière-plan sur la machine d'origine ; un checkpoint plus
tardif (voire final) sera probablement meilleur. Si tu veux le dernier point
sauvegardé au moment où tu lis ceci, demande.
