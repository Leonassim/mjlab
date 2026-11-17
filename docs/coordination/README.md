# Coordination entre les deux machines d'entraînement

Deux sessions Claude Code entraînent en parallèle, chacune sur sa machine. Elles communiquent **uniquement par ce
dépôt** (`Leonassim/mjlab`, branche `main`). Léo a demandé ce partage le 2026-11-17.

| Machine | GPU | Ce qui y est installé | Journal |
|---|---|---|---|
| `tardis` | RTX 2080 Ti 11 Go (12-14 s par itération) | mjlab, superbuild mc_rtc, mc_mujoco, rl_controller | [`tardis.md`](tardis.md) |
| `rtx4080` | RTX 4080 SUPER 16 Go (plus rapide) | mjlab (mc_mujoco à voir) | [`rtx4080.md`](rtx4080.md) |

## Répartition

- **`rtx4080` : le professeur**, c'est-à-dire la politique escaliers qui lit le scan exact. C'est elle qui fixe le
  plafond de tout le reste, et ses runs sont les plus longs, donc ils vont sur la machine la plus rapide.
  - **D'abord, des marches plus hautes que 6 cm**, depuis `checkpoints/rhps1/s50e_50hz_15000`. Runs `hs*`.
  - **Ensuite**, la qualité de la marche à plat et l'immobilité à l'arrêt (objectifs de Léo, novembre 2026).
- **`tardis` : l'élève vision, le banc mc_mujoco et le contrôleur.**
  - L'élève vision : il lit la carte embarquée de `vision_map.py` (`RHPS1_VIS=1`) et part du meilleur professeur.
    Runs `vis*`.
  - Le banc mc_mujoco de 12 essais, **pour les deux machines**.
  - Le côté rl_controller : slot 50 Hz, tête tenue, interface carte et pose.
  - Le PC du robot.
- **Quand le professeur progresse nettement** (palier 12/12 à une hauteur de plus), `tardis` repart du nouveau
  professeur pour l'élève.

## Protocole

1. **Chaque machine n'écrit que dans son propre journal** (`<machine>.md`). Elle y note ses runs, ses résultats et
   ses demandes à l'autre. On ne modifie jamais le journal de l'autre, ce qui évite les conflits.
2. **`git pull --rebase` avant chaque push.** Petits commits, messages en anglais, **jamais de `Co-Authored-By`**.
3. **Code partagé** (`stairs.py`, `vision_map.py`, scripts) :
   - toute nouveauté passe derrière une variable d'environnement, **défaut inchangé**, pour ne pas changer en douce
     les runs de l'autre machine ;
   - chaque changement s'annonce dans son journal, avec le hash du commit.
4. **Checkpoints** : ne pousser que les références et les candidats, pas chaque modèle (environ 11 Mo chacun).
   - Dossier `checkpoints/rhps1/<nom>/` avec `model_*.pt`, `agent.yaml`, `env.yaml`, `policy.onnx` et, si besoin,
     le fichier de niveaux `*_lvl.pt`.
   - Ajouter une ligne dans `checkpoints/rhps1/README.md`.
5. **Demander un banc mc_mujoco** à `tardis` :
   - pousser l'ONNX dans `checkpoints/candidates/<tag>_<itération>.onnx`, avec `scripts/stairs/export.sh`, qui
     vérifie la parité ;
   - ajouter dans son journal une ligne `BANC: <fichier> [hauteurs]`, par exemple `BANC: hs1_17000.onnx 0.04-0.09` ;
   - `tardis` répond dans `tardis.md` avec la ligne `RELBENCH`.
6. **Nommer chaque run** : tag, nom du dossier (date), identifiant wandb (projet `mjlab`) et checkpoint de départ.

## Règles de Léo (pour les deux sessions)

- **Langue et accords**
  - Répondre à Léo en français, court ; code et commits en anglais.
  - Son accord est nécessaire avant d'installer un slot dans rl_controller, de changer une version épinglée ou
    d'utiliser une référence d'imitation (RL pur).
- **À chaque lancement**
  - wandb et vidéos de 3 s ; `train_s50.sh` le fait.
  - `--env.scene.num-envs` explicite (3584 dans la recette).
  - Surveillance armée sur le bon PID.
  - GPU jamais inactif.
- **Shell et processus**
  - Pas d'attente basée sur `pgrep -f`.
  - Lancer en `bash` et non en `zsh`.
  - Ne pas arrêter un processus qu'on n'a pas lancé.
  - Ne pas envoyer de vidéos à Léo.
- **Méthode**
  - Une seule modification à la fois par rapport à la configuration qui marche.
  - Juger dans mc_mujoco à 50 Hz (`RLQP_BENCH_POLICY_STEP=0.02`).
  - Ne pas conclure sur un seul run pour de petits écarts.
- **Pour démarrer** : lire [`scripts/stairs/README.md`](../../scripts/stairs/README.md), puis
  [`checkpoints/rhps1/README.md`](../../checkpoints/rhps1/README.md). L'historique et les leçons sont dans
  [`docs/historique_escaliers.md`](../historique_escaliers.md).
