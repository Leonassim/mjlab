# abl15 — checkpoints à tester (temporaire)

Run `2026-08-05_11-17-44`, 15000 itérations, 4096 envs. Première politique
entraînée avec l'observation de couple brut + la pénalité `raw_torque_peak`
en `log1p`.

**Dossier temporaire, à supprimer une fois le test fait** — 22 Mo de poids dans
un dépôt public.

## Quoi tester

| fichier | itération | pourquoi |
|---|---|---|
| `model_10050.pt` | 10050 | **candidat principal** — meilleur compromis couple/stabilité |
| `model_14999.pt` | 14999 | final, pour comparer |
| `model_10050.onnx` | 10050 | export ONNX du candidat, pour mc_rtc (slot 1) |
| `2026-08-05_11-17-44.onnx` | 14999 | export ONNX du final (celui que l'entraînement écrit) |

Pour exporter l'ONNX d'un autre checkpoint :
`uv run python scripts/tools/export_onnx.py 2026-08-05_11-17-44 model_XXXX.pt`

## Comment

Recopier dans l'arborescence de logs attendue par `play` :

```
mkdir -p logs/rsl_rl/rhps1_velocity/2026-08-05_11-17-44
cp tmp/abl15_checkpoints/*.pt tmp/abl15_checkpoints/*.onnx logs/rsl_rl/rhps1_velocity/2026-08-05_11-17-44/
cp -r tmp/abl15_checkpoints/params logs/rsl_rl/rhps1_velocity/2026-08-05_11-17-44/
```

Puis (avec `MJLAB_RHPS1_XML` pointé sur ta copie locale des xmls/meshes) :

```
uv run play Mjlab-Velocity-Flat-RHPS1-Play --agent.load-run 2026-08-05_11-17-44 --agent.load-checkpoint model_10050.pt
```

## Pourquoi 10050 et pas le final

Les gains s'arrêtent vers l'itération 10000 pendant que les chutes montent :

| | 10000 | 12500 | 15000 |
|---|---|---|---|
| `raw_torque_peak_mean` | 1,046 | 1,026 | 1,019 |
| `over_limit_fraction` | 0,389 | 0,381 | 0,377 |
| `fell_down` | 0,049 | 0,206 | 0,143 |
| `upright` | 2,760 | 2,669 | 2,749 |

2,6 % de couple gagné sur les 5000 dernières itérations pour trois fois plus de
chutes. Le poids de curriculum est figé à −1.50 depuis l'itération 8000, donc ce
n'est pas un transitoire de palier.

Sur l'ensemble du run en revanche c'est net : pic moyen 2,67 → 1,019, fraction
au-dessus de la limite 0,610 → 0,377, demande `kp` 1,50 → 0,605.

## Ce qui n'est pas vérifié

- `raw_torque_peak_max` oscille encore entre 120 et 190 en fin de run :
  l'objectif du `log1p` — écraser la queue de distribution — n'est **pas**
  démontré.
- Tous ces chiffres sont sous bruit d'exploration (facteur 3,3 mesuré sur
  abl7). Le rollout déterministe reste à faire.
- Le ratio de la pénalité est `|τ|/effort_limit` avec `effort_limit` = 70 N·m au
  genou, soit 3,3× le continu réel de 21,4 N·m. « 1,0 » en simu est donc très
  au-dessus du réel.
- L'observation est en **566 dims (V5)** ; le C++ de `rl_controller`
  (`utils.cpp`, cas 0) est encore en **126 dims (V3)**. L'ONNX n'est pas
  déployable tel quel sur le robot.
