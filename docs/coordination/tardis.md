# Journal `tardis` (RTX 2080 Ti)

Les entrées les plus récentes sont en haut.

## 2026-11-18 (2 h) : chaîne vision dans mc_mujoco, deux défauts corrigés ; vis2 lancé

Banc habituel (`relbench`, 12 essais à 4, 5 et 6 cm), avec vis1 17500 sur le slot 23 et `RLQP_VISION=1`.

| Entrée de vis1 | sans chute | palier |
|---|---|---|
| scan exact | 12/12 | 12/12 |
| carte, avant correctifs (MCWaiko 6D) | 5/12 | 5/12 |
| carte, vraie pose, avant correctif des empreintes | 8/12 | 10/12 |
| carte, vraie pose, **après correctif des empreintes** | **12/12** | **12/12** |
| carte, MCWaiko (z compris) | 6/12 | 8/12 |
| carte, x/y/lacet de MCWaiko + **z par odométrie des jambes** | 9/12 | 12/12 |
| idem + recalage de la carte à chaque contact (`--align`) | 10/12 | 10/12 |

- **Défaut 1, mon code (corrigé dans `rl_controller` 19fa3c5) : les empreintes.** Le capteur de cheville lit -10 à
  -19 N sur un pied en vol, à cause de son poids et de son inertie. Le seuil |Fz| > 10 N marquait donc le pied en
  vol comme appuyé 91 % du temps, et son empreinte rehaussait la carte derrière le robot d'environ 3 cm. Le seuil
  passe à Fz > 100 N. Avec la vraie pose, la carte vaut maintenant le scan exact.
  - **À vérifier chez vous** : `vision_map.py` prend la force de contact (> 10 N), qui est nulle en vol. Pas de
    problème donc, mais un futur modèle de capteur de cheville aurait le même piège.
- **Défaut 2 : la hauteur de MCWaiko.** Son inclinaison est juste à 0,1° près, mais son z 6D surestime les montées
  (+5 à +13 cm pour 12 cm montés). La vision prend maintenant le z de l'odométrie des jambes : la semelle la plus
  chargée garde sa hauteur, ancrée **sur son coin le plus bas**. Le centre de la semelle remonte à chaque attaque
  du talon, d'environ 1 cm par pas.
- **Ce qu'il reste** : la dérive en z sur 1 s de l'odométrie des jambes, rejouée hors ligne avec la vraie
  orientation, vaut 0,6 cm en médiane et 1,5 cm au p95, toujours vers le haut. Le modèle d'entraînement de
  `vision_map` ne prévoyait que 0,2 cm de marche aléatoire en 1 s et ±10 % de la montée.
- **vis2** :
  - départ : vis1b 18000 (vis1 reprise après le disque plein) ;
  - **seule modification** : `RHPS1_VIS_ODO_ZRW=0.012` (au lieu de 0,002) et `RHPS1_VIS_ODO_ZSCALE=0.3` (au lieu de
    0,1) ;
  - dossier `2026-11-18_01-44-00`, portes `vis_gate_loop`.
- **Pour l'élève par l'image (depth1)** : il n'a pas besoin de cette odométrie. En revanche, l'`obs_format` 13 côté
  contrôleur devra garder l'état caché du GRU entre les pas et le remettre à zéro à l'armement. Je le ferai quand
  vous pousserez un candidat.

## 2026-11-18 (1 h) : BANC hs2_19000 avec porte-à-faux, chaîne vision dans mc_mujoco, prochaine tâche

### Banc hs2_19000

`scripts/stairs/ohbench.sh` reprend le banc habituel : 12 essais à 4, 5 et 6 cm, avec des départs à 2, 3, 4 et 5 s.
Deux changements :
- chaque essai dure 30 s, pour laisser le temps de descendre ;
- il ajoute `scripts/stairs/overhang.py`, qui reconstruit la cinématique depuis le log et lit chaque appui chargé
  0,1 s après le contact.

Pour chaque appui, deux mesures :
- **semelle hors marche** : la part de la face inférieure de la semelle mc_mujoco (9 × 3 points) qui se trouve plus
  d'une demi-contremarche sous la marche d'appui ;
- **à cheval** : 3 rayons sur 6 ou plus hors de la marche, avec les 6 rayons de `stairs_soles`. C'est le critère de
  `stairs_gait.py`, donc comparable à vos mesures mjlab.

| | sans chute | palier | descente | semelle hors marche (descente) | à cheval (descente) | à cheval (montée) |
|---|---|---|---|---|---|---|
| hs2_19000 | 12/12 | 12/12 | 4/12 | 13 % | **8 %** (143 appuis) | 2 % (200) |
| s50e 15000 | 12/12 | 12/12 | 7/12 | 20 % | 14 % (170) | 16 % (190) |

- **Dans mc_mujoco, hs2 réduit nettement les appuis à cheval** : de 14 à 8 % en descente, et de 16 à 2 % en montée.
  mjlab ne voyait pas cet effet. Le défaut signalé par Léo est donc bien réel dans mc_mujoco, et hs2 le corrige en
  partie.
- **Prix à surveiller** : hs2 descend moins souvent en 30 s (4/12 contre 7/12). Il hésite plus longtemps au bord du
  palier. C'est le risque de « descente figée » que vous suiviez.

### Chaîne vision dans mc_mujoco (`tardis`)

- **La chaîne marche.** Elle est sur `rl_controller` `real-robot-safe` :
  - l'`obs_format` 12, qui lit le même scan que le 11, mais pris dans la carte embarquée ;
  - la mémoire partagée `src/vision_link.h` ;
  - le slot 23 = vis1 17500, à 50 Hz, tête tenue à 25° ;
  - `scripts/vision/rhps1_vision.py --source sim`, lancé par `RLQP_VISION=1 scripts/mc_mujoco_stairs.sh`.
  - Mesures : carte âgée de 0,10 à 0,15 s, tête tenue, erreur moyenne du scan sous 1 cm.
- **Piège trouvé : MCWaiko sans `leggedOdometry` n'estime pas la position.** Sa position est alors l'ancre du robot
  de commande, qui ne marche jamais dans le contrôleur RL. Elle reste au point de départ sur 5,5 m de marche, et le
  z ne suit pas les marches. La config du contrôleur passe en `odometryType: 6D`.
  - Sur le slot 18, les entrées de la politique sont identiques pendant 1 s après l'armement, et le chemin est le
    même après 30 s.
  - Avec la 6D, la dérive en x n'est que de quelques centimètres sur 4 m, mais **le z dérive de +5 à +13 cm sur
    12 cm montés**.
- **Résultats de vis1 17500**, banc habituel (relbench) :

  | Entrée | sans chute |
  |---|---|
  | carte embarquée | 5/12 |
  | scan exact | **12/12**, comme s50e 15000 |

  L'élève vaut donc le professeur. C'est la carte de mc_mujoco qui le fait tomber, alors que les portes mjlab
  (`vismoke`) ne lui coûtaient presque rien.
  - Suspect principal : la dérive en z de MCWaiko. L'entraînement ne modélise que 10 % d'erreur en z.
  - Test en cours : la même carte, construite et lue avec la vraie pose.

### Prochaine tâche proposée à `rtx4080` (accord de Léo, 2026-11-18) : élève par l'image de profondeur

**Idée** : plus de carte ni d'odométrie. C'est la famille d'Agarwal 2022, d'*Extreme Parkour* (Cheng 2023) et de
*Humanoid Parkour* (Zhuang 2024). Une petite image de profondeur de la tête passe dans un CNN, puis dans un GRU, puis
dans la politique. La mémoire retient les marches vues, que la tête ne voit plus sous les pieds.

- **Départ** :
  - professeur `s50e_50hz_15000`, la référence mc_mujoco ; ou hs2 si vous le jugez meilleur, mais une seule
    modification à la fois ;
  - distillation DAgger avec `rsl_rl` 5.5.2 (`algorithms/distillation.py`, `runners/distillation_runner.py`). Elle a
    `CNNModel` et `RNNModel`, mais pas de modèle CNN + GRU : à écrire dans `rl_ext.py`, avec le CNN en encodeur
    devant le GRU.
- **Image** :
  - caméra de `vision_map.py` : Azure sur `HEAD_P_LINK`, `CAM_POS` / `CAM_RPY`, tête tenue à 25° avec
    `RHPS1_VIS=1` et la surcharge de `hold_action.py` ;
  - motif sténopé d'environ 64 × 48 dans le cône WFOV de 120°, distances écrêtées entre 0,25 et 2,88 m ;
  - le robot occulte (groupe de géométries 3, comme dans `vision_map`) ;
  - images à 25 Hz avec 80 ms de retard : garder la dernière image.
- **Dégradations à l'entraînement** :
  - bruit de distance de 1 cm ;
  - erreur de montage de 1° et 1 cm par épisode ;
  - environ 5 % de pixels manquants ;
  - pixels parasites mélangeant avant et arrière-plan aux bords des marches, comme sur la vraie Azure.
- **Déploiement visé** : le CNN tourne sur le Jetson et envoie le latent (32 à 64 nombres) au PC de contrôle par le
  Wi-Fi, ou bien tout tourne sur le PC de contrôle. Il faut donc un petit encodeur, sous 1 ms sur CPU.
- **Tests** :
  - rollouts mjlab déterministes, avec des départs proches et lointains ;
  - puis le banc mc_mujoco chez `tardis`, qui ajoutera la source d'image à `rhps1_vision.py` et l'`obs_format` 13.

### Outils

- `relbench.sh` et `mc_mujoco_suite.sh` effacent leurs logs mc_rtc après le résumé (`KEEP_LOGS=1` pour les
  garder). `~/logs/mc_rtc` avait atteint 298 Go et rempli le disque de `tardis` ; vis1 en est mort, puis a repris
  en vis1b.

## 2026-11-17 (20 h) : BANC hs1_17000 (6 à 9 cm), et référence s50e 15000

Banc mc_mujoco à 50 Hz, 4 départs de commande par hauteur (2, 3, 4 et 5 s), 25 s. « Palier » : bassin au-delà de
1,04 m ; « descente » : au-delà de 3,5 m.

| Hauteur | hs1_17000 | s50e 15000 |
|---|---|---|
| 6 cm | palier 4/4, **descente 0/4** (s'arrête vers x = 2,2-3,4 m) | palier 4/4, descente 4/4 (banc du 17, 4-6 cm) |
| 7 cm | palier 4/4, descente 0/4 | palier 4/4, descente 0/4 |
| 8 cm | palier 2/4, 1 chute | palier 4/4 (banc partiel) |
| 9 cm | palier 0/4, 3 chutes ; reste devant la première marche | palier 0/4, 4 chutes à la première contremarche |

`RELBENCH hs1_17000_h6-9: no-fall 12/16, platform 10/16, down 0/16 | knee>=105 4.8% | leg vmax 5.6`

- **hs1 a perdu la descente à 6 cm** : il s'arrête sur le palier ou au bord, alors que s50e 15000 descendait
  4 fois sur 4. Ça ressemble au piège déjà vu (« les descentes depuis le haut se figent : rester immobile coûte
  moins que tomber »). Les marches plus hautes rendent la descente plus risquée, et la politique apprend à rester
  en haut.
- **En montée**, hs1 gagne un peu à 9 cm (moins de chutes, mais il ne monte pas) ; 8 cm reste fragile.
- **Suggestion** : avant de relever encore le plafond, vérifier les descentes dans l'entraînement (taux
  d'arrivée en descente par niveau, part d'immobilité au bord). Combiné à la remarque de Léo sur l'appui du pied
  en descente, la descente est le point à travailler.

## 2026-11-17 (19 h) : retours de Léo après ses tests mc_mujoco, pour `rtx4080`

Léo trouve que le robot marche plutôt bien. Il relève deux défauts, à intégrer dans l'entraînement du professeur
(une modification à la fois) :

1. **En descente, le pied se pose trop au milieu**, à cheval sur le nez de marche : il manque de l'appui sous la
   semelle.
   - Piste : mesurer d'abord la part de semelle en appui à chaque pose en descente, avec les rayons
     `stairs_soles` (grille 3 × 2 sous la cheville). Voir `_support` et `stairs_foothold` dans `stairs.py`.
   - Puis renforcer le coût de pied en porte-à-faux pour les descentes seulement, puisque la montée lui va.
   - Le giron fait 27 cm pour une semelle de 23 cm : il y a la place de poser toute la semelle.
2. **Pas de contrôle en lacet** (Léo y tient). Le latéral n'a pas été vérifié, mais il n'a jamais été entraîné non
   plus : les commandes d'entraînement sont avant ou arrêt seulement.
   - Piste : des commandes de lacet (et de latéral) en direction (-1, 0, +1), comme l'avant, d'abord sur le plat et
     pendant l'approche, puis sur les marches.
   - Attention aux deux pièges déjà vus :
     - ap5 : une récompense positive de cap payait le fait de rester debout et figeait la politique ;
     - ap6 : la commande « tourner + avancer » (HEAD_CMD) la figeait aussi.
   - Utiliser un coût sur l'erreur de vitesse de lacet, pas une récompense, et garder le suivi d'avance tel quel.
   - Côté contrôleur, `command_as_sign` envoie déjà le lacet du joystick en -1, 0, +1. Je vérifierai que la
     composante passe bien dans l'observation.

Dès que tu as un candidat, pousse-le avec une ligne `BANC:`. J'ajouterai au banc une mesure du porte-à-faux en
descente.

## 2026-11-17 (17 h)

- **Réponse sur le taux d'apprentissage de s50e (`ama9seoz`)** : oui, il a glissé au plancher.

  | Itérations | Médiane | Temps au plancher (1e-5) |
  |---|---|---|
  | 3000-5000 | 5,1e-5 | 1 % |
  | 5000-7000 | 2,3e-5 | 13 % |
  | 7000-9000 | 1,5e-5 | 26 % |
  | 9000-12000 | 1e-5 | 71 % |
  | 12000-15700 | 1e-5 | 100 % |

  Ça colle avec le plateau. vis1 est aussi à 100 % au plancher. La piste « mesurer la KL réelle puis lever le
  plancher » te revient (professeur) ; je reprendrai ce qui marche pour l'élève.
- **vis1 model_15500**, 500 itérations après la reprise, sur la carte embarquée : départ proche 31 arrivées et
  4 chutes (le professeur en faisait 9), départ lointain 25 arrivées et 8 chutes (pareil que le professeur).
- **BANC hs1_17000 (6 à 9 cm) et s50e 15000 (7 à 9 cm)** : lancés à 17 h, résultats ici vers 17 h 45.

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
