# Escaliers RHPS1 : entraîner, suivre, tester

Tout ce qu'il faut pour reproduire l'entraînement escaliers à 50 Hz (run s50 / s50e, novembre 2026) sur une autre
machine. L'historique complet du sujet est dans [`docs/historique_escaliers.md`](../../docs/historique_escaliers.md),
les checkpoints de référence dans [`checkpoints/rhps1/`](../../checkpoints/rhps1/README.md).

## Mise en place sur un nouvel ordinateur

1. **mjlab** (ce dépôt, fork `Leonassim/mjlab`, branche `main`) :
   ```bash
   git clone git@github.com:Leonassim/mjlab.git ~/mjlab-rhps1
   cd ~/mjlab-rhps1 && uv sync --extra cu130   # torch 2.14 CUDA 13 ; --extra cu126 pour un pilote CUDA 12
   ```
   Testé sur RTX 2080 Ti (sm_75) avec le pilote 580.
2. **Modèle RHPS1** : le XML et les maillages ne sont pas dans ce dépôt (exclus par `.gitignore`). Ils viennent du
   dépôt privé AIST [`isri-aist/rhps1_mj_description`](https://github.com/isri-aist/rhps1_mj_description)
   (dossiers `xml/` et `meshes/`, identiques à ceux utilisés ici). Au choix :
   ```bash
   git clone git@github.com:isri-aist/rhps1_mj_description.git ~/src/rhps1_mj_description
   export MJLAB_RHPS1_XML=~/src/rhps1_mj_description/xml/RHPS1main.xml   # le meshdir ../meshes suit
   # ou bien des liens dans l'arborescence attendue par rhps1_constants.py :
   ln -s ~/src/rhps1_mj_description/xml    src/mjlab/asset_zoo/robots/RHPS1/xmls
   ln -s ~/src/rhps1_mj_description/meshes src/mjlab/asset_zoo/robots/RHPS1/meshes
   ```
3. **Simulation mc_mujoco** (seulement pour les tests en simulation complète) : superbuild mc_rtc
   (`Leonassim/mc-rtc-superbuild`, branche `real-robot-safe`, qui épingle `rl_controller` et applique le patch
   mc_mujoco des drivers RHPS1). Le script de test est `~/src/rl_controller/scripts/mc_mujoco_stairs.sh`.
4. **wandb** : `wandb login` (projet `mjlab`). Les vidéos de 3 s sont enregistrées toutes les 250 itérations.

## Entraîner

```bash
scripts/stairs/train_s50.sh s50               # depuis zéro (comme s50)
scripts/stairs/train_s50.sh s50e <run> <ckpt> # reprise (comme s50e), ex. 2026-11-14_22-21-43 model_3000.pt
DRY=1 scripts/stairs/train_s50.sh test        # affiche la configuration sans lancer
```

- La recette complète est `s50e.env` : les 130 réglages `RHPS1_*` du run s50e, relevés sur le processus en cours.
  En départ de zéro, le lanceur réactive les rampes (régularisations, cadence, enveloppe BWC, contrainte de
  saturation, approche) sur 72 000 pas et met l'entropie à 0,002. En reprise, entropie 0 et rampes figées.
- Avec l'entropie 0,002, l'écart-type de la politique est monté de 1,0 à 1,75 en 3000 itérations. Reprendre en
  entropie 0 après 1000 à 3000 itérations (c'est ce qu'a fait s50e).
- Journaux, fichiers de niveaux du curriculum et résultats des bancs : `logs/stairs_runs/` (`RHPS1_RUNS` pour
  changer). Checkpoints : `logs/rsl_rl/rhps1_tracking/<run>/`.
- Coût : 13,8 s par itération sur 2080 Ti (3584 envs, 24 décisions à 50 Hz = 0,48 s simulée par env).

## Suivre

```bash
scripts/stairs/gate_loop.sh <run> <tag> <first_it> &   # toutes les 500 its : export + banc mc_mujoco + rollouts mjlab
.venv/bin/python scripts/stairs/tb_trend.py <run> "Policy/mean_std,Episode_Termination/stairs_fall"  # courbes
```

- `gate.txt` : 12 essais mc_mujoco (4/5/6 cm × départ 2/3/4/5 s après l'armement), palier atteint, chute,
  couple genou ≥ 105 N·m, vitesse articulaire max. C'est le juge principal.
- `detcheck.txt` : rollouts déterministes mjlab (32 envs, 24 s), départ proche et départ lointain : arrivées,
  chutes, vitesse d'avance.
- Le banc mjlab de chutes (`kneestat.py`) est très sévère (friction tirée jusqu'à 0,3) : un écart de moins de
  ~7 chutes sur 48 est du bruit. Juger en mc_mujoco.

## Tester dans mc_mujoco

```bash
RLQP_BENCH_POLICY_PATH=<policy.onnx> RLQP_BENCH_POLICY_STEP=0.02 ~/src/rl_controller/scripts/mc_mujoco_stairs.sh 0.05
```

- Le script crée l'escalier (3 marches, palier de 1,6 m, première contremarche 0,30 m devant le bassin) et charge
  le slot 22 avec ce fichier. Armer depuis l'interface, puis commande avant au joystick.
- **`RLQP_BENCH_POLICY_STEP=0.02` est obligatoire pour une politique 50 Hz** : le contrôleur l'interroge alors
  toutes les 20 ms et maintient la consigne entre deux décisions. Sans lui, elle tourne à 200 Hz et le test est faux.
  Les politiques 200 Hz (rl51, rl54, ap7, slot 22) se testent avec `0.005` ou sans la variable.
- Sans écran : `scripts/stairs/relbench.sh <onnx> <label>` (12 essais, une ligne de résumé).
- Pour le vrai robot, il faudra un slot dédié avec `policy_step_size: 0.02` (pas encore créé).

## Outils d'analyse (mjlab, CPU)

`scripts/stairs/probe.sh <script.py> <args> [-- VAR=VAL ...]` lance un outil avec la recette :

| Outil | Ce qu'il mesure |
|---|---|
| `apsmoke.py <run> <ckpt>` | rollout déterministe, positions/cap/commande, ligne `SUMMARY` |
| `stairs_terms.py <run> <ckpt> <label>` | contribution de chaque terme de récompense par seconde (`STOCH=1` avec bruit) |
| `kneestat.py <run> <ckpt> <label>` | chutes, angle du genou en appui, longueur de pas par hauteur |
| `ftrace.py <run> <ckpt>` | trace des 2-3 s avant chaque chute |
| `swingstat.py <run> <ckpt> <label>` | durées d'appui simple et risque de chute associé |
| `evprobe.py <run> <ckpt>` | variance expliquée du critique |
| `s50smoke.py` | contrôle de câblage (pas de temps, proportions d'envs, NaN, coûts par seconde) |
| `stairs_bench.py`, `stairs_gait.py` | bancs par hauteur (arrivées, chutes, impacts, posture, saturation) |

`history/` garde les scripts de toutes les étapes (`chain_rl*.sh`, `chain_ap*.sh`, `chain_s50*.sh`) tels qu'ils
ont tourné, avec leur configuration complète en tête de fichier ; les chemins y pointent vers l'ancien dossier
temporaire, ils servent de mémoire, pas à être relancés tels quels.
