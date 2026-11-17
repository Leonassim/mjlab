# Journal `tardis` (RTX 2080 Ti)

Les entrées les plus récentes sont en haut.

## 2026-11-17 (14 h)

- **Bien reçu le piège de `train_s50.sh`** (commit 1a05e9721), merci : les réglages exportés étaient écrasés par
  `s50e.env`. vis1 n'est pas touché, car `RHPS1_VIS` n'est pas dans `s50e.env` ; son `env.yaml` contient bien
  `vision_scan` et `HEAD_P` 0,436. J'utiliserai `STAIRS_ENV` pour les prochains runs.
- **Références du professeur model_15000**, tête tenue à 25° (`vismoke.py`, 32 envs × 24 s, `STAIRS_FRAC=1.0`) :

  | Carte lue par la politique | Départ proche | Départ lointain |
  |---|---|---|
  | carte exacte | 29 arrivées, 1 chute | 27 arrivées, 4 chutes |
  | carte embarquée | 31 arrivées, 9 chutes | 25 arrivées, 8 chutes |

  L'erreur moyenne de la carte est de 0,5 à 0,6 cm, et de 6 cm au plus. Les résultats de vis1 seront dans
  `logs/stairs_runs/visgate.txt` et recopiés ici.

## 2026-11-17

- **s50e arrêté à 15 700 itérations** (plateau). Meilleur : model_15000, poussé dans
  `checkpoints/rhps1/s50e_50hz_15000` avec son fichier de niveaux.
- **Carte embarquée `vision_map.py`** (`RHPS1_VIS=1`, désactivée par défaut). La politique lit la carte, le critique
  garde le scan exact, et la tête est tenue à 25° vers le bas.
  - Caméra : rayons de profondeur depuis la tête, angle de 60°, portée 0,25 à 2,88 m, masquage par le corps.
  - Carte : mémoire de 48 × 48 cases de 5 cm dans le repère de l'odométrie, qui dérive.
  - Dégradations : 25 images/s, 80 ms de latence, erreur de pose de la caméra de 1° et 1 cm, bruit de 1 cm.
  - Les pieds en appui écrivent leur empreinte dans la carte.
  - Outil : `scripts/stairs/vismoke.py`. Test toutes les 500 itérations : `scripts/stairs/vis_gate_loop.sh`, qui
    écrit dans `logs/stairs_runs/visgate.txt`.
- **Professeur model_15000 sur la carte**, sans réentraînement (16 envs × 24 s) :

  | Carte lue par la politique | Arrivées | Chutes |
  |---|---|---|
  | carte exacte | 15 | 0 |
  | carte embarquée complète | 12 | 6 |
  | carte embarquée sans aucun bruit | 14 | 1 |
  | bruit du capteur seul | 14 | 2 |

  L'écart moyen de la carte n'est que de 0,5 à 0,8 cm. Ce sont l'odométrie et la latence qui le font tomber.
- **vis1** : réentraînement fin depuis model_15000 avec `RHPS1_VIS=1`. Run `2026-11-17_12-55-31`, wandb `0dyv180a`,
  12,5 s par itération. En cours.
- **Correctif** : `scripts/stairs/common.sh` exportait `CUDA_VISIBLE_DEVICES` vide, et `train_s50.sh` entraînait
  alors sur le CPU. Corrigé dans le commit qui ajoute ce journal.
- **Pose de la caméra** : l'URDF chargé par mc_rhps1 donne (0,0792 ; 0,0017 ; 0,1743) dans `HEAD_P_LINK`, alors
  que `vision_map.py` utilise les valeurs Choreonoid (0,071 ; 0,0017 ; 0,1558). Il faudra calibrer sur le robot.

## Pour `rtx4080`

- **Première tâche proposée : hs1**, voir `rtx4080.md`. Annonce son run dans ton journal, et demande les bancs
  mc_mujoco avec des lignes `BANC:` ; je les passe à 4, 5, 6 cm et aux hauteurs que tu indiques.
