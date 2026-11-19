# Plan : professeur omnidirectionnel depuis zéro (o50), 2026-11-19

## Demande de Léo

Réentraîner le professeur **depuis zéro**, avec des récompenses et un curriculum pensés pour les nouveaux
mouvements :
- marche arrière, pas de côté et rotation sur place, pour manœuvrer le robot ;
- les escaliers restent **uniquement en marche avant** : jamais de montée de côté ou en tournant ;
- les garde-fous d'équilibre du slot 18 dès le départ : pas d'impact fort, pas de mouvement brusque, pas de
  balancement, et un arrêt propre sans glissement.

Motif : les fine-tunes depuis s50e (t1 à t4) ne bougent pas. Une politique convergée en marche avant n'absorbe
pas une nouvelle façon de bouger. C'était déjà la leçon de T12 (§11, objectifs_et_decisions.md).

## Constats qui fixent la conception

- s50e est en RL pur depuis zéro, sans imitation. La recette s50 apprend vite (épisodes de 2,7 à 17,8 s en
  1 h). Les résultats sont bons en mc_mujoco dès 3500 itérations, et 12/12 à 10500.
- t3 (vy et lacet en fine-tune) a appris à **ne plus tomber en restant sur place** (0 arrivée, vitesse
  latérale ≈ 0). La récompense de progression vers un but positionnel ne suffit pas pour le latéral et la
  rotation.
- t4 (coût de roulis -150 en fine-tune) : roulis p95 inchangé à 4,1° (slot 18 : 2,2°).
- L'arrêt de s50e glisse (13 cm), fait 2 poses correctives et n'est jamais immobile. La récompense d'arrêt ne
  dit rien de l'alignement des pieds.
- Le contrôleur envoie le **signe** des 3 composantes (`command_as_sign`), sur 27 états possibles. Une
  diagonale à la manette donne deux composantes non nulles, et elle doit avoir été vue à l'entraînement.

## Conception proposée

1. **Base** : la recette s50e (50 Hz, entropie 0, robustesse complète, rampes de régularisation sur 3000
   itérations, chute rampée). Mêmes escaliers, même curriculum de hauteur, `MODE_FIXED`.
2. **Interface inchangée** : observation = signe de (vx, vy, lacet), donc aucune modification de rl_controller.
   Vitesses visées par axe :
   - avant : 0,12 m/s (comme aujourd'hui) ;
   - arrière : 0,08 m/s ;
   - côté : 0,08 m/s ;
   - lacet : 0,3 rad/s.
3. **Répartition des épisodes** :
   - escaliers 60 %, en marche avant seulement, comme s50e ;
   - plat 40 %, sous forme de **séquences de commandes** : 3 à 5 segments de 1,5 à 4 s tirés parmi avant,
     arrière, ±côté, ±lacet, avant+lacet et zéro. Les segments à zéro entraînent l'arrêt et le redémarrage à
     chaque épisode.
4. **Récompense des modes plats** : suivi de vitesse dans le repère du corps, avec un noyau exp sur (vx, vy,
   ωz) autour des vitesses visées. Le but positionnel et l'arrivée restent pour les escaliers.
5. **Garde-fous dès le départ** (rampés avec les autres régularisations) :
   - enveloppe du slot 18 (vitesses articulaires, cheville, pied en l'air) ;
   - `ONE_LEG_T` ;
   - coûts d'atterrissage et d'impact ;
   - cadence ;
   - roulis sans zone morte (`W_ROLL`) ;
   - posture d'arrêt, pieds côte à côte et CoM au milieu (`W_STOPPOSE`) ;
   - glissement à l'arrêt (`W_STILLSLIP`).
6. **Risque connu** : « vivre coûte plus cher que tomber » (rl3-rl7). On le traite comme dans s50 : chute peu
   coûteuse au début puis rampée, régularisations rampées, et suivi du budget par seconde au lancement.

## Portes de validation

| Banc | Critère |
|---|---|
| omni_eval (avec marche arrière et diagonales) | vitesse ≥ 70 % de la cible sur chaque axe, 0 chute, avance parasite ≤ 0,03 m/s |
| stop_eval | roulis p95 ≤ 2,5°, glissement net ≤ 2 cm (slot 18 : 1,6 ; s50e : 5,7 dans mc_mujoco), 0 pose après l'arrêt, immobile < 1,5 s |
| tgate (relbench) | ≥ s50e sur toutes les lignes (moteur 0,53 et 0,45, avec arrêts) |
| mc_mujoco (tardis) | relbench 12/12, banc arrêt, banc omni, essai manette |

## Coût et calendrier

- 1 run sur tout le GPU : t3 et t4 sont arrêtés au lancement.
- Lecture à environ 3000 itérations (~12 h), décision à environ 6000 (~1 jour).
- s50e reste la politique de référence pour les escaliers tant que o50 ne la bat pas.

## Questions ouvertes (avis tardis)

- Diagonales : faut-il les entraîner toutes (27 états), ou faire choisir un seul axe au contrôleur ? La
  deuxième option demande une modification de rl_controller, donc l'accord de Léo.
- Faut-il ajouter, en marche avant, une référence BWC indexée par la phase (P1 du §11) pour retrouver le
  roulis du slot 18, ou le coût de roulis suffit-il ?
- Les vitesses visées conviennent-elles pour la manœuvre ?
