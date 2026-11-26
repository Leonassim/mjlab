# Marche RL sur plat dans toutes les directions : état au 2026-11-25

Objectif (Léo, 2026-11-24) : une politique RL classique sur plat qui suit avant, arrière, côté et lacet,
avec des pas levés, stable, peu d'oscillation, arrêt propre. Le slot 18 sert d'étalon, pas de référence à copier.
Ensuite : escaliers avec curriculum (étape B), puis vision.

## Base

- Tâche `Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start`, 50 Hz (`RHPS1_DECIMATION=8`), PPO rsl_rl.
- Recettes : fichiers `.env` dans `scripts/stairs/`, lancés par `scripts/stairs/train_s50.sh <tag> [<run> <ckpt>]`
  avec `STAIRS_ENV=<recette>`, `NUM_ENVS`, `ENT_COEF`, `EXTRA_ARGS` (options CLI de train).
- Code des termes : `src/mjlab/tasks/tracking/config/rhps1/stairs.py` (séquences de commandes OSEQ, récompenses,
  état). Actionneurs et échelle d'action : `src/mjlab/asset_zoo/robots/RHPS1/rhps1_constants.py` et
  `env_cfgs.py` (`RHPS1_ACTION_SCALE_MULT=10`, jambes x1,5). Écart-type initial : `rl_cfg.py`
  (`RHPS1_INIT_STD`, défaut 1,0).
- Options de reprise dans `src/mjlab/scripts/train.py` :
  - `MJLAB_LOAD_ACTOR_ONLY=1` : acteur seul, critique et optimiseur neufs ;
  - `MJLAB_CRITIC_WARMUP=N` : acteur gelé N mises à jour, lr tenu à 1e-3 ;
  - `MJLAB_RESUME_LR` : lr imposé quand l'acteur apprend (une reprise complète restaure le lr de l'optimiseur
    par-dessus l'option CLI).
- Bancs (`scripts/stairs/flat_gate.sh <tag> <run> <premier>`, toutes les 500 its) :
  - `oseq_eval.py` : 8 commandes imposées pendant 6 s puis arrêt ; ratio vitesse/cible par axe, fuite, roulis,
    glissement à l'arrêt ;
  - `stop_eval.py` : marche avant puis arrêt ; roulis, cadence, simple appui, levée de pied, genou de balancier,
    glissement, décalage des pieds. Corrigé le 2026-11-25 : il héritait de `OSEQ_P` de la recette et testait
    des séquences aléatoires ;
  - `fall_probe_oseq.py <run> <ckpt> vx,vy,wz` : tangage, hauteur, charges, levée jusqu'à la chute.

## Recette actuelle (lignée fw → gb → gi)

`fw.env` : recette o50/fa sur plat pur (`FRAC=0`), cadence `W_CAD 20`, garde-fous du slot 18 (roulis, hike,
enveloppes BWC, atterrissage, impact), chute -200, 20 % de tuiles à but en avant (`W_PROG 60`, arrivée 50,
arrêt -20) et 80 % de séquences de commandes (`OSEQ_P 0.8`, suivi `W_OTRACK 24` sur le déplacement net en 1 s).

`gb.env` = fw plus les corrections de la nuit :

| Variable | Rôle |
|---|---|
| `CRITIC_CMD=1` | le critique lit la commande, la vitesse suivie et le minuteur d'arrêt (il ne voyait que la commande du clip BWC) |
| `OSEQ_N 3,4`, `OSEQ_T 5,9` | segments de 5 à 9 s (avant : 1,5 à 4 s puis zéro jusqu'à la fin, soit plus de la moitié du temps « reste debout ») |
| `OSEQ_STALL 0.3` | coût d'arrêt sur les séquences sous 30 % de la vitesse demandée |
| `W_OPROG 60` | cliquet de progression par segment dans la direction commandée (lacet compté 0,27 m/rad), plafonné à la vitesse cible : la version séquence de la progression qui a appris l'avant à fw |

`gi.env` = gb, repris de la marche avant de fw : gh (gf 250, lr fixe 1e-4) puis gi (gh 750, lr fixe 3e-4).
gf = acteur seul de fw 3000, chauffe du critique 100 mises à jour, `MJLAB_RESUME_LR 1e-5`.

## Chiffres des bancs

oseq_eval : ratio vitesse/cible (chutes sur 8).

| Ckpt | Avant | Arrière | Gauche | Droite | Lacet+ | Lacet- | Avant+lacet | Avant+gauche |
|---|---|---|---|---|---|---|---|---|
| fw 3000 / gf 250 | 0,92 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gh 1000 | 0,88 (1) | 0 | 0,02 | 0,05 | 0,28 | 0,46 | 0,03 / 0,55 | 0,22 / 0,39 |
| gi 1000 | 0,79 (2) | 0,17 (3) | 0,34 (1) | 0,47 (4) | 0,51 (3) | 0,20 (1) | -0,03 / 0,48 (5) | 0,53 / 0,86 (1) |

stop_eval : marche avant puis arrêt, 32 envs.

| Ckpt | Debout | Roulis p95 | Pas | Simple appui | Levée | Glissement arrêt | Décalage pieds |
|---|---|---|---|---|---|---|---|
| fw 3000 | 29/32 | 2,8° | 0,42 s | 84 % | 1,7 cm | 9,4 cm | 16,8 cm |
| gh 1000 | 28/32 | 2,6° | 0,44 s | 87 % | 1,8 cm | 3,0 cm | 3,2 cm |
| gi 1000 | 28/32 | 2,3° | 0,46 s | 91 % | 1,9 cm | 10,5 cm | 16,6 cm |
| slot 18 (mc_mujoco) | — | 2,2° | 1,03 s | 96 % | 5,6 cm | 1,6 cm | — |

Copies : `checkpoints/rhps1/gh_1000/` et `checkpoints/rhps1/gi_1000/` (non versionnées).

## Runs en cours

- **gi2**, run `2026-11-25_09-51-27` : gi repris de gi 1000, lr fixe 3e-4, 1792 envs. Bancs à partir de 1500.
- **gj**, run `2026-11-25_09-53-07` : gb depuis zéro, `RHPS1_INIT_STD=0.3`, 1792 envs. Bancs toutes les 500 its.

Lancement :

```bash
MJLAB_RESUME_LR=3e-4 EXTRA_ARGS="--agent.algorithm.schedule fixed --agent.algorithm.learning-rate 3e-4" \
  STAIRS_ENV=scripts/stairs/gi.env ENT_COEF=0 NUM_ENVS=1792 scripts/stairs/train_s50.sh gi2 2026-11-25_09-05-26 model_1000.pt
STAIRS_ENV=scripts/stairs/gj.env ENT_COEF=0 NUM_ENVS=1792 scripts/stairs/train_s50.sh gj
```

## Ce qui n'a pas marché (journal : docs/coordination/rtx4080.md, 2026-11-24 et 25)

- **Depuis zéro, toutes directions** (fo à fv, gb) : la politique glisse (levée 0,1 à 0,6 cm, 2 à 13 % de
  simple appui), tourne ou se décale en glissant, et tombe ou se fige en avant. Toutes les récompenses de vol
  ou de pose se gagnaient avec des levées de 1 mm ; `AIR_MIN_H` et `stairs_step` (pose payée si la semelle est
  montée de 3 cm) sont arrivés tard.
- **Lignée minimale fm** (garde-fous coupés, glissement -4) : glisse ; la chute rampée (`FALL_S0 0.15`) coûtait
  environ 20 et pencher en avant puis tomber rapportait. Note : `train_s50.sh` impose `LAND_RAMP_OFF=0`
  depuis zéro, donc la rampe de chute de la recette ne joue qu'en reprise.
- **Reprises de fw** (fz, gc, gf, gg) : ajoutent zéro axe. Deux causes :
  - sans `CRITIC_CMD`, le critique ne distingue pas les commandes ;
  - le lr adaptatif reste au plancher 1e-5, et toute commande autre que l'avant pur fige le marcheur.
  Il faut un lr fixe à 1e-4 ou plus (gh, gi).
- **Reprise acteur seul** : une seule mise à jour avec un critique neuf a cassé la marche (12/16 chutes).
  Pendant la chauffe, le lr adaptatif est monté à 1e-2 (KL nul), ce qui a cassé ge au dégel.

## Analyse en cours : servo raide

RHPS1 : Kp de 10 000 à 20 000 N·m/rad, couple max / Kp = 2 à 7 mrad (G1 : 0,3 à 2,2 rad), inertie 1,0 partout
en simulation (réel : cheville 0,016 à 0,048, genou 0,42). Avec un écart-type de 1,0, le bruit d'exploration
vaut 3 à 6° par articulation toutes les 20 ms, soit 10 à 30 fois la plage linéaire : le moteur sature en
permanence, la politique réduit son bruit et se fige ou glisse.

Pistes :
1. écart-type initial plus faible (gj, en cours) ;
2. filtre passe-bas sur les consignes ;
3. bruit corrélé dans le temps ;
4. action en vitesse ou en incrément ;
5. compliance ou admittance au pied comme le BWC ;
6. vraies inerties avec un actionneur de position implicite, à aligner avec mc_mujoco.

## Défauts connus de gi

- Chutes de 1 à 4 sur 8 selon l'axe.
- Pas rapides et bas : 0,46 s et 1,9 cm, contre 1,03 s et 5,6 cm au slot 18.
- Glissement à l'arrêt de 10 cm et pieds décalés.
- Arrière faible : 0,17.

## Mise à jour 2026-11-25 17 h 40 : meilleur candidat gyaw 4000

`checkpoints/rhps1/gyaw_4000` (ONNX inclus), lignée gi2 → gk (genou) → gyaw (lacet crédité en simple appui seulement,
`OSEQ_YAW_SS`). Premier candidat qui fait des pas sur tous les axes, sans chute, dans mc_mujoco (banc tardis) :
lacet en simple appui 77-80 % (gs/gk : 35-53 %), roulis p95 1,1-1,6°, glissement à l'arrêt 2-4 cm.
Écarts au slot 18 : pas 0,41-0,46 s (1,03), levée 1,6-2,0 cm (5,6-6,4), genou de balancier +6..+9° (+22).

Leçons de la journée :
- un coût sur un levier fait baisser ce levier (hike -150 → genou inchangé ; coût de lacet de semelle → pivot plus lent),
  il faut payer directement le bon levier ou créditer la tâche seulement quand elle est faite comme voulu
  (`OSEQ_YAW_SS`) ;
- la cadence (`W_CAD 20`, `CAD_C 0.2`) payait presque autant les pas de 0,5 s que ceux de 1 s : c'est elle qui fixait
  les pas courts (bilan par seconde de tardis). `CAD_C 0.45` (gcad) allonge les pas à 0,66 s, levée 2,9 cm,
  genou +10, mais dégrade le lacet ; gcy = gcad 4000 + `OSEQ_YAW_SS` combine les deux (en cours) ;
- ni la saturation (diagnostic tardis), ni le plafond de vitesse du genou, ni l'enveloppe d'inclinaison de semelle ne
  bloquent le genou (`swing_tilt_probe.py`).

## Mise à jour 2026-11-27 matin : candidat robot gimpCT 10000

`checkpoints/rhps1/gimpCT_10000` (ONNX inclus). Lignée gimp 6250 → gimpP2 → gimpS → gimpC → gimpCT, un seul écart à chaque étape :

| Étape | Écart | Pourquoi |
|---|---|---|
| gimpP2 | dérive de posture des jambes vers q0 (`W_PDRIFT -2`, `PDRIFT_FREE 0.02`, `PDRIFT_STD 0.05`) | chutes en arrière sur le robot depuis un repos accroupi, bassin penché en arrière |
| gimpS | `PDRIFT_REF=stance` : référence = posture debout réelle du contrôleur quand la commande est nulle | le slot 18 ne revient pas à q0 mais à cette posture (genou 29,6 au lieu de 35,6) |
| gimpC | `STOP_COM_X 0.045` : CdM visé 4,5 cm devant les chevilles à l'arrêt | slot 18 : +4,6 cm ; la pose d'arrêt payait un CdM au-dessus des chevilles, donc le recul |
| gimpCT | `W_TD -10`, `TD_VMAX 0.13` | gimpC dérivait vers des impacts plus forts après 9000 |

mc_mujoco (tardis) : repos CdM +4,7 cm, IMU +2,4° (slot 18 +4,1°) ; diagonale arrière-gauche tangage min -3,1/-3,0°,
recul droit avec CdM réel -3,7 cm -2,6° ; Fz p90 avant 779 N, arrière 881 N, lacet 905 N (à surveiller).

Bancs ajoutés : `stop_eval` ligne « repos » (bassin vs q0, jambes vs q0 et vs posture debout réelle, CdM devant les
chevilles) ; `oseq_eval` couple de roulis de hanche, foulée et Fz/vz de pose par axe.

Problème ouvert : le gel des axes. Toute bascule de commandes (gcur, gcurr, vall) ou mélange depuis zéro (gmix, vomni)
donne une politique qui ne marche que sur l'avant pur et reste debout sinon (coût d'arrêt < coût de chute). Les seuls
marcheurs multi-axes viennent de la lignée reprise fw → gh → gi (lr fixe, CRITIC_CMD, OSEQ_YAW_SS).

Mode vitesse : apprend vite depuis zéro seulement avec le haut du corps libre ; les bras se corrigent ensuite
(vcurpA2 3000 : haut du corps à moins de 9° de q0). Même gel des axes que la position.
