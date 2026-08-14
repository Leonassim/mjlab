# Checkpoint de test

Un seul checkpoint, remplacé à chaque fois — pas un historique. Le but est de
pouvoir le relire sur une autre machine sans traîner des dizaines de fichiers
de 7 Mo dans l'historique git.

`2026-08-12_20-36-28_model_14999.pt` — run 2026-08-12_20-36-28, itération
finale 14999/15000 (run terminé). Le `.onnx` du même point est joint pour
tester directement sous mc_rtc sans réexporter. `env.yaml` est la config
effective de ce run.

## Ce que ce run apporte

Première policy entraînée **avec le filtre de PostureTask modélisé**
(`posture_task_stiffness = 1600` dans l'actionneur), c'est-à-dire avec les
~25 ms de retard que le QP impose au déploiement et que l'entraînement
ignorait jusqu'ici. Plus quatre correctifs mesurés contre le log robot réel
du 2026-08-10 et une randomisation de domaine élargie :

- `alpha_range` de `link_inertia` corrigé — c'est un **logarithme**
  (`exp(2*alpha)`), l'ancienne valeur simulait un robot de 350 kg
- `encoder_noise` ramené de 0.001 à 5e-5 : mesuré contre le robot, 0.001
  mettait 21× trop de bruit sur `joint_vel`
- biais capteur constant par épisode (vitesse ±0.05, gravité ±0.01)
- friction de sol **par environnement** (avant : un seul sol partagé par les
  4096 envs, donc pas randomisé du tout), poussées de récupération, inertie
  et centre de masse randomisés sur tous les corps

## ATTENTION — ce checkpoint ne marche pas correctement

Quatre mesures indépendantes, qui concordent à 1–2 % de suivi.

`scripts/tools/why_video_walks.py`, commande injectée dans `_update_command` et
vérifiée, 64 envs, 0.2 m/s sur 5 s (attendu 1.00 m) :

| condition | p50 | p90 | suivi p50 |
|---|---|---|---|
| `play=True` + déterministe | +0.023 m | +0.116 m | 2 % |
| `play=True` + stochastique | +0.010 m | +0.117 m | 1 % |
| `play=False` (config des vidéos) + déterministe | +0.009 m | +0.117 m | 1 % |
| `play=False` + stochastique | −0.012 m | +0.102 m | −1 % |

`scripts/tools/natural_command_tracking.py`, aucune commande forcée du tout —
curriculum actif, commandes échantillonnées, bruit d'exploration, soit
exactement ce que `VideoRecorder` filme pour wandb. Déplacement projeté sur la
direction demandée, envs « immobiles » exclus : **p50 à 2 %, 4 envs sur 124
au-dessus de 50 %.**

Enfin, sans passer par aucun script : la boucle d'entraînement elle-même
rapporte `Metrics/twist/error_vel_xy = 0.322` en fin de run, contre 0.146–0.154
pour la policy 0, pour un curriculum arrivé au même endroit (±0.3 m/s). L'erreur
de suivi est aussi grande que la commande.

Le comportement observé en mc_mujoco (pas de pas en avant) est donc **fidèle**
à ce que fait la policy, ce n'est pas un problème de portage.

### Deux pièges de mesure rencontrés ici, tous deux corrigés

**Le bruit d'exploration n'explique PAS l'écart avec les vidéos.** C'était
l'hypothèse initiale et elle est fausse : reproduire la config filmée, bruit
compris, donne le même 1–2 %. Ce que la vidéo montre est une cadence, pas une
translation — caméra qui suit le robot sur un plan, un pas sur place ressemble
à une marche.

**Forcer la commande après `env.step()` ne force rien.**
`command_manager.compute()` tourne *à l'intérieur* de `step`, juste avant
`observation_manager.compute()` : la valeur écrite après le retour est réécrite
avant que l'observation soit construite. Le seul point d'injection correct est
`_update_command` lui-même. Relire `cmd.command` juste après l'avoir écrit est
un contrôle tautologique — il faut le lire en début de boucle, donc après le
`compute()` du pas précédent.

## Métriques finales, comparées à la policy 0 (la seule validée sur robot)

| | policy 0 | ce run |
|---|---|---|
| `impact_vel` | ~-0.19 | **-0.474** |
| `fell_down` | 0.000 | **0.645** |
| `torque_limit_ratio_mean` | 0.39–0.46 | **0.913** |
| `sole_height_p50` | n/a (métrique buguée à l'époque) | 0.9 mm |

## Différences de plant avec la policy 0

Vérifiées paramètre par paramètre — la configuration de déploiement est
correcte, ce ne sont pas des erreurs de recopie :

| | policy 0 | ce run |
|---|---|---|
| `effort_limit` genou | 100 N·m | **70 N·m** |
| `action_scale` genou | 0.0075 rad/unité | **0.00525** |
| filtre PostureTask | absent | K=1600 |
| projection de faisabilité couple | absente | active (ratio 1.0) |
| EMA vitesse, `velLim`, décimation, dt | identiques (0.8 / 10 / 2 / 0.0025) | |

Le rapport `scale/effort` est conservé (7.5e-5), donc ce n'est pas un
déréglage — mais l'autorité absolue du genou est 30 % plus faible des deux
côtés à la fois.

## Reproduire la mesure

```bash
uv run play Mjlab-Velocity-Flat-RHPS1 \
  --checkpoint-file test_checkpoint/2026-08-12_20-36-28_model_14999.pt
```

Pour le chiffre du tableau plutôt que l'impression visuelle, il faut un
rollout déterministe avec la commande **forcée à chaque pas** et le curriculum
désactivé (`cfg.curriculum = {}`) — sinon le curriculum réécrit les plages de
commande et on mesure autre chose. C'est l'erreur qui a fausse mes deux
premières tentatives.
