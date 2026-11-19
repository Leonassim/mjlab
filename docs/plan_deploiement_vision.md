# Plan : une politique escaliers vision déployable (rtx4080, 2026-11-19)

Demande : une politique qui aide à monter et descendre les marches avec la caméra, et qui est déployable comme le
slot 18. Concrètement :
- pas d'impact fort ni de mouvement brusque ;
- jamais de chute ;
- immobile à commande nulle ;
- la vision ne la fait jamais tomber et ne provoque pas de geste brusque.

On peut régler les récompenses et entraîner un nouveau professeur, à condition de mesurer et d'analyser à chaque
étape. Les références sont `docs/objectifs_et_decisions.md` (O1 à O4, D1 à D6) et `docs/vision_safety.md`.

## Constat de départ (mesures)

- **Le professeur s50e 15000 ne passe pas les critères escaliers de Léo** sur le banc à hauteurs fixées
  (`stairs_gait.py`, 8 envs par case, déterministe) :
  - chutes à 4,2, 5,1 et 6,0 cm, en montée comme en descente ;
  - atterrissages : vitesse p90 de 0,20 à 0,40 m/s, 8 à 24 % au-dessus de 0,21 m/s ; force p90 de 920 à 1580 N ;
  - pas parfois très courts (période p10 de 0,10 à 0,23 s), c'est-à-dire des mouvements brusques ;
  - saturation du genou de 2 à 4 % du temps.

  hs2 19000 fait à peu près pareil. Pourtant, s50e fait 12/12 dans mc_mujoco (`tardis`) et 10/12 ici. **Il faut d'abord
  savoir lequel des bancs dit vrai** (le banc à hauteurs fixées a déjà été jugé trop sévère), puis améliorer le
  professeur : l'élève ne fait jamais mieux que lui en douceur.
- **Élèves image** :
  - au banc dur, à appareil égal, ils chutent davantage que le professeur, sauf après le PPO (rl2) ;
  - un passant devant la caméra est le seul danger mesuré, neutralisé par le garde-fou qui masque l'image
    (`docs/vision_safety.md`) ;
  - depth8 est le premier élève avec la vraie caméra.

## La porte de déploiement

Un checkpoint est candidat seulement s'il passe tout. Les seuils marqués « réf. » sont fixés après la mesure du
slot 18 et du professeur (phase 0), puis écrits ici.

| | Critère | Mesure | Seuil |
|---|---|---|---|
| P1 | Jamais tomber | géométrie du relbench (giron 0,27, 4 et 5 cm), montée et descente, avec et sans arrêt en pleine marche, entraînement moteur fixé à 0,53 et à 0,45 ; randomisation complète ; plat ; commande nulle | 0 chute |
| P2 | Impacts faibles | `stairs_gait.py` (escaliers) et `impact_walking.py` (plat) : vitesse d'atterrissage p90 et p99, force p90 | ≤ réf. slot 18 sur le plat, ≤ 0,21 m/s p90 sur les marches |
| P3 | Pas de mouvement brusque | variation d'action p99, accélération articulaire p99, période de pas p10, inclinaison du tronc p95 | ≤ réf. slot 18 ; période p10 ≥ 0,3 s |
| P4 | Couples faisables | saturation par articulation, pic de genou | ≤ réf. slot 18 (O1) |
| P5 | Immobile à commande nulle | pas parasites, pieds à plat (D6) | 0 pas |
| P6 | Sûr face à la vision | banc de robustesse avec garde-fou (passant, image coupée, figée) ; variation d'action pendant une panne | pas de chute en plus ; pas plus brusque qu'en marche normale |
| P7 | Simulation complète | relbench mc_mujoco 12/12 (4, 5, 6 cm) ; banc vision obligatoire de Léo (boîte de 25 cm à commande nulle, carte coupée, carte figée) | 12/12 et 0 chute |
| P8 | Écart sim-réel | `depth_real_check.py` sur des images réelles ; reconstruction du scan par l'élève sur ces images | écart dans le bruit simulé ; boîte et marches retrouvées |

## Phases

**Phase 0 : mesurer (en cours, environ un jour)**
1. Passer le slot 18, s50e, depth8 et rl3 aux mêmes outils pour remplir les références de P2 à P5.
2. Réconcilier le banc à hauteurs fixées avec mc_mujoco : rejouer dans mjlab les 12 essais du relbench (mêmes
   hauteurs, même départ) pour savoir si les chutes du banc fixé sont réelles.
3. Écrire `deploy_gate.py` : un checkpoint en entrée, une ligne par critère en sortie, PASS ou FAIL, pour le
   professeur comme pour l'élève.

**Phase 1 : un professeur déployable** (si la phase 0 confirme ses défauts)
- Partir de s50e 15000. Changer une seule chose à la fois, d'abord ce qui vise P2 et P3 :
  - coût sur la vitesse d'atterrissage au-dessus de 0,21 m/s ;
  - coût sur la force de contact au-delà d'environ 1,5 fois le poids ;
  - variation d'action et accélération articulaire ;
  - période de pas minimale.
- Porte complète tous les 500 itérations. On garde un changement seulement si la porte progresse sans recul ailleurs,
  et on l'écrit dans le journal.
- Repartir de zéro seulement si le réglage fin plafonne. Raison à donner : un défaut que le réglage fin ne corrige
  pas.

**Phase 2 : l'élève image** (depuis le meilleur professeur)
- **Imitation (DAgger)** :
  - vraie caméra et vrai flux ;
  - dégradations par épisode ;
  - pannes à l'entraînement : coupures de 0,2 à 3 s, image figée, passants, blocs fantômes ;
  - l'image vide du garde-fou, pour que l'élève connaisse ce que le contrôleur lui donnera.
- **Puis PPO** avec les mêmes récompenses que le professeur : il réduit les chutes de 28 à 41 % (rl1, rl2).

**Phase 3 : sécurité et réel**
- P6 avec garde-fou, P7 dans mc_mujoco (format d'observation de l'élève, avec `tardis` et Léo), P8 sur les
  enregistrements réels.
- Garde-fou dans rl_controller, avec l'accord de Léo.

**Phase 4 : robot**, seulement quand P1 à P8 passent. D'abord commande nulle, puis le plat, puis une marche, sous
harnais.

## Suivi

- **Pour chaque run** :
  - wandb avec des vidéos de 3 s ;
  - `Loss/learning_rate` surveillé (leçon de rl40-46) ;
  - porte automatique tous les 500 itérations, une ligne par checkpoint dans `logs/stairs_runs/deploygate.txt` ;
  - résumé dans le journal `rtx4080`.
- **Une seule modification par run**, avec la raison et le verdict écrits dans le journal.
- **Le GPU ne reste jamais libre** : le prochain run est préparé avant que le précédent finisse.

## Hauteurs cibles

**Décision de l'utilisateur (2026-11-19) : les premiers essais réels se font sur des obstacles et des marches jusqu'à
5 cm.** La porte P1 à P7 se juge donc de 0 à 5 cm, montée et descente. Le relbench mc_mujoco 4-5-6 cm reste la référence
commune avec `tardis` ; une chute à 6 cm n'y bloque pas un candidat, mais elle se note.

## Plancher moteur (2026-11-19)

`RHPS1_DRIVE_SCALE` : la seule mesure vaut 0,53 (chevilles, balancement à 0,97 Hz). Le plancher de 0,35 de
l'entraînement est une marge, jamais mesurée (`tardis`). **La porte se juge à 0,45** ; 0,35 se note sans bloquer.

## Questions ouvertes

- **Garde-fou dans rl_controller** : à décider avec Léo.
