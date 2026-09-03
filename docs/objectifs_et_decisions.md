# RHPS1 — objectifs, contraintes, et journal des décisions

**Ce fichier fait foi.** Toute décision de configuration doit s'y référer :
citer l'objectif visé, et après mesure, écrire le verdict. Une piste rejetée ne
se retente pas sans que la section 6 dise ce qui a changé depuis.

---

## 1. Objectifs de Léo

Par ordre de priorité, tels qu'énoncés.

### 1.1 Ne doit jamais casser

| | critère | référence |
|---|---|---|
| O1 | **Couples faisables** — l'action doit être admissible telle quelle | policy 0 : `satleg` 0.0101, `satup` 0.0027 |
| O2 | **Vitesses d'impact faibles** | policy 0 : 0.19 au balayage |
| O3 | **Ne jamais tomber**, quelle que soit la commande | 0.0000 sur 11 commandes |
| O4 | **Haut du corps calme**, la tête en particulier | — |

### 1.2 Défauts à corriger

| | défaut | cible |
|---|---|---|
| D1 | Lever de pied insuffisant | **3 à 5 cm** au-dessus du sol |
| D2 | Se tient trop en arrière, sur les talons | charge avant/arrière équilibrée |
| D3 | Marche arrière impossible sans tomber | se juge **sur robot uniquement** |
| D4 | Latéral mauvais | lié aux contraintes de self-collision du QP |
| D5 | Pas trop rapides | air **0.5 s**, double appui **0.5 s**, période ~1 s |
| D6 | Contacts sales | pas de déséquilibre entre les 4 points ; **à l'arrêt, immobile sur deux pieds à plat** |

### 1.3 Méthode imposée

- **M1** — Mesurer sur une politique existante avant de lancer un entraînement.
- **M2** — Monitorer, tester, ajuster, régler. Ne pas laisser tourner 20 h.
- **M3** — Le GPU ne reste jamais libre.
- **M4** — wandb et vidéos 3 s à chaque lancement.
- **M5** — Une heuristique ou un critère, **pas de l'imitation learning**. Repli
  envisagé seulement en dernier recours : donner les trajectoires BWC en entrée
  pour guider l'exploration.
- **M6** — Réponses courtes.

---

## 2. Contraintes dures

**C1 — mc_mujoco n'écrête pas le couple.** La projection qui borne
`|tau| <= effort_limit` est sautée au déploiement : sous QP elle épingle la
cible à 0.0018 rad sur CROTCH_Y et rend le robot mou
(`NewRLQPController.cpp:380`). C'est le `torch.clamp` de mjlab qui absorbe
l'excédent **à l'entraînement seulement**. Donc l'action doit être admissible
par elle-même, et l'ordre de grandeur acceptable est celui de la policy 0.

**C2 — les métriques d'entraînement portent le bruit d'exploration.** Elles
surestiment la saturation d'un facteur ~10. Seul le balayage déterministe
décrit ce que le robot exécutera. (`comshift` : 0.15 à l'entraînement, 0.0115
au balayage.)

**C3 — les limites de vitesse articulaire sont plates à 8.0 rad/s** des deux
côtés, alors que le robot a de vraies limites par articulation (tête 4,
poitrine 6). Écart connu, non corrigé. Le corriger se fait **à l'entraînement**,
pas au déploiement, sous peine de recréer une divergence.

**C4 — reprise à configuration identique = effondrement.** Sur cinq reprises,
les trois avec changement de config ont récupéré (suivi 2.87, 3.02, 3.08), les
deux sans se sont figées debout sur les talons (1.26, 0.53). Mécanisme inconnu,
corrélation 5/5.

**C5 — `RewardManager` divise par `step_dt`.** Un terme payé à l'événement
devient un tarif proportionnel au **nombre** d'événements : un bonus par pose
récompense d'en faire plus, un coût par pose d'en faire moins.

**C6 — une cible clampée se pose AU-DESSUS de la mesure.** En dessous elle
sature et le gradient est nul ; trop au-dessus le noyau s'écrase et le gradient
disparaît aussi. Viser 1.3 à 1.5 fois la mesure.

**C7 — RÈGLE ABSOLUE. Aucun coût attaché à l'atterrissage ne peut être
augmenté**, ni par le poids, ni par le plafond, ni par le seuil. La demande est
toujours satisfaite en n'atterrissant plus.

Cinq occurrences : `cbal` (double appui 0.94), `capture` (0.88), `freevel` seul
(0.94), `flat_touchdown`, et `foot_swing_height` à −3.0 **malgré une horloge à
3.16/s**.

**Le plafond de `impact_vel` à 0.20 avait été compté comme une sixième : c'est
faux.** Le checkpoint de reprise (`22-19-33 model_3300`) avait été pris 150
itérations après une reprise, alors que la run planait déjà — track 1.11, air
2.73 s, chutes 0.10. La run `descent` qui a suivi, avec le plafond rendu à 0.45
et un terme entièrement différent, a reproduit la MÊME trajectoire à trois
décimales près. L'effondrement venait de la reprise, pas du barème.

Le levier reste un coût ou un gain payé **pendant le vol**, par seconde.
`swing_height_bonus_dense` (hauteur) et `descent_speed_cost` (vitesse de
descente) sont les deux formes qui marchent.

**Corollaire ajouté à C4 : ne jamais prendre un checkpoint sans vérifier l'état
de sa run à cette itération.** Trois checkpoints de la campagne — `3899`,
`6900`, `3300` — ont été pris en plein transitoire et ont contaminé tout ce qui
en repartait. Le seul point de reprise fiable est un checkpoint dont le
balayage déterministe a été fait.

---

## 3. Références chiffrées

### Policy 0 — la seule avec du temps robot

Balayage déterministe, 1024 environnements, 11 commandes.

| critère | valeur |
|---|---|
| `satleg` / `satup` | 0.0101 / 0.0027 |
| chutes | 0.0000 |
| impact | 0.1919 |
| lever de pied | 0.0051 |
| pieds à plat | 0.0035 |

### BaselineWalkingController — la démarche visée

Log `2026-08-27-17-47-32` (genou à 82 N·m ; les deux runs suivants plafonnent à
45.0 exactement, ce sont ceux que Léo a écrêtés).

| grandeur | valeur |
|---|---|
| appui simple | 0.765 s |
| double appui | 0.135 s |
| période | ~0.90 s |
| clearance du pied | 6.3 cm |
| CoM latéral | 11.4 cm |
| offset CoM ↔ pied d'appui | 7.0 à 7.7 cm |
| hauteur de CoM | 0.96 m |
| profil de déport (% appui) | 1.6 → 4.7 → 1.8 cm |
| profil de vol (% vol) | pic 6.3 cm à 30 % |
| placement du pied vs point de capture | sagittal 1.5 cm, latéral 7.2 cm |

---

## 4. État courant

Politique déployée en index 4 : `comshift` (`rhps1_comshift_it5099.onnx`,
commit `978eb83`), 510 dims, `obs_format 5`.

Politique en cours : horloge + `steplen` ×4, run `2026-08-30_20-39-24`.
**Non déployable** : 530 dims, il faut un `obs_format` de plus et répliquer
l'horloge dans le contrôleur C++.

---

## 5. Journal des décisions

Format : objectif visé → ce qui a été fait → mesure → verdict.

| # | vise | changement | mesure | verdict |
|---|---|---|---|---|
| 1 | sim-to-real | `hist5+mirror+masscom+prox` | couple 0.332 vs 0.345, chutes 0, impact 0.112 vs 0.132 | **GARDÉ** — gratuit |
| 2 | D6 | `cbal` 0.5 (répartition de force, bonus/s d'appui) | double appui 0.13 → 0.94, chutes 0 → 22 %, **evenness 0.49 → 0.18** | **REJETÉ** — la grandeur payée empire : minimum local, pas un dosage (C7) |
| 3 | D1 | `swt` (cible `foot_swing_height` 0.15 → 0.05) | pic +39 %, rien de dégradé | **GARDÉ** |
| 4 | D1 | `fclr` (retrait de `foot_clearance`, qui taxait la vitesse horizontale du pied) | clearance +33 %, pas +11 %, impact +9 % | **GARDÉ** — arbitrage assumé |
| 5 | — | consolidation sans changement | suivi 2.92 → 1.10, talons 0.79 | **REJETÉ** — c'est C4 |
| 6 | D5 | `slowstep` (cible distance SOUS la mesure) | période 0.204 → 0.164 | **REJETÉ** — la moitié saturée devient une prime par pas (C5) |
| 7 | D5 | `slowstep` cible au-dessus | période 0.204 → 0.181 | **REJETÉ** — signe corrigé, effet insuffisant |
| 8 | D5 | `freevel` (desserrer le suivi de vitesse ×10) | période inchangée | **REJETÉ** — le tracking ne tenait pas la cadence |
| 9 | D5 | `freevel` seul, sans `steplen` | double appui 0.94, ne marche plus | **REJETÉ** — rien ne paie plus l'avancement |
| 10 | D5/D1 | `comshift` (offset CoM ↔ pied, cible 7 cm) | impact **0.1900** (record), couple 0.0115, période 0.223 | **GARDÉ** — meilleure politique déployée |
| 11 | D5 | `comprof` (profil de déport indexé sur la phase) | déport monte, **période descend**, couple 0.42 | **REJETÉ** — infirme l'hypothèse du pendule comme *levier* |
| 12 | D1 | terme sur la forme du vol | RL pique à 0.167 du vol, BWC à 0.30 | **ÉCARTÉ** — mais la mesure était contaminée par le broutement (voir 18) : la vraie valeur est 0.327, soit celle du BWC. Conclusion inchangée, raison corrigée |
| 13 | D5 | hauteur de CoM constante | — | **ÉCARTÉ** — artefact du LIPM, décision de Léo |
| 14 | D1/D5 | `capture` (pénalité de placement sur le point de capture) | double appui 0.88, suivi 0.99 | **REJETÉ** — C7, prédit dans sa propre docstring et lancé quand même |
| 15 | D5 | **horloge de démarche** (Siekmann, phase en observation) | période 0.204 → 0.45, air 0.15 → 0.36, impact 0.058 | **GARDÉ** — premier levier qui prescrit au lieu de récompenser |
| 16 | D5 | horloge poids 2.0 → 1.0 | — | **ANNULÉ** — coupé sur un point bas alors que le suivi remontait |
| 18 | D1 | **porte sur le temps de vol** dans `split_feet_swing_height` : ne compter un atterrissage que si le pied a volé plus de 0.05 s | `peak_height_mean` 0.0009 → 0.0076 (vérité mesurée 0.011), `peak_time_frac` 0.167 → 0.327 (BWC 0.30) | **CORRECTIF** — le terme était dominé par le broutement du solveur |
| 17 | D5 | `steplen` ×4 (0.5 → 2.0) sous horloge | période 0.45 → **0.72**, air **0.497**, foulée 0.011 → 0.022, clearance 0.0077 → 0.0101 | **GARDÉ** |

### Balayage déterministe de 15+17, `2026-08-30_20-39-24` model_5250

1024 environnements, 11 commandes, seuils resserrés à 0.03 (contrainte C1).

| critère | mesure | seuil | vs policy 0 | |
|---|---|---|---|---|
| impact faible | **0.1485** | 0.160 | 0.1919 | **OK — jamais atteint avant** |
| couples faisables | 0.0127 | 0.030 | 0.0101 | OK, même ordre |
| couples haut du corps | 0.0000 | 0.030 | 0.0027 | OK |
| ne jamais tomber | 0.0029 | 0.010 | 0.0000 | OK |
| pieds à plat | 0.0173 | 0.050 | 0.0035 | OK |
| lever de pied | 0.0046 | 0.030 | 0.0051 | **ÉCHEC** |

Période mesurée 0.70 à 0.88 s selon la commande, contre 0.90 pour le BWC.
**Cinq objectifs sur six**, dont O2 pour la première fois de la campagne.

À l'arrêt, commande nulle forcée : chutes 0.0000, double appui 0.9423,
inclinaison de semelle 0.0113 rad, couple 0.1234. D6 tenu sauf sur la
répartition de charge (`evenness` 0.2504) et un pas résiduel toutes les ~3 s.

---

| 19 | D1 | **porte sur le temps de vol** + **division par `step_dt`** dans `foot_swing_height` | `peak_height_mean` 0.0009 → 0.0150 (vérité 0.011–0.015), terme ×31 | **CORRECTIF** — le broutement de contact facturait l'erreur maximale à chaque micro-contact, et masquait le défaut C5 en le déclenchant 200× trop souvent |
| 20 | D1 | `foot_swing_height` −1.0 → −3.0, pari que l'horloge neutralise C7 | air_time 0.49 → **1.55 s**, chutes 0.008 → 0.83, couple 0.41 → 0.46 | **REJETÉ** — C7 vaut **même sous horloge** : la politique préfère payer l'horloge (3.16/s) plutôt que la pénalité de pose. Toute la famille « coût à l'atterrissage » est éliminée pour D1 |
| 21 | — | retour à −1.0 depuis le checkpoint d'origine | air_time 5.76 s, clearance 0.0029, chutes 0.56 | **REJETÉ** — c'est C4 : revenir à la config d'un checkpoint EST une reprise à configuration identique. On n'annule pas un échec par un retour de poids |
| 22 | D1 | **`swingbonus`** — hauteur payée **par seconde de vol**, aucune pénalité de pose (`foot_swing_height` à 0.0) | clearance **0.0217** contre 0.0156 au mieux, période 0.704, air 0.492, chutes 0.0006, sat jambes 0.162 | **GARDÉ** — première réponse franche à D1. Immunisé à C7 par construction : ne pas se poser ne rapporte rien de plus |

### Run propre `2026-08-31_15-11-13`, pile complète

`hist5+mirror+masscom+prox+swt+fclr+comshift+clock+steplen+swingbonus`, depuis
zéro — quatre reprises consécutives depuis `model_6900` ayant donné des
transitoires dont aucune n'est revenue.

| | it 467 | it 931 | it 1397 | it 1863 | sans bonus |
|---|---|---|---|---|---|
| clearance | 0.0091 | 0.0146 | 0.0174 | **0.0217** | 0.0058 |
| période | 0.271 | 0.390 | 0.466 | 0.704 | 0.217 |
| air | 0.199 | 0.289 | 0.386 | 0.492 | — |
| chutes | 0.000 | 0.001 | 0.000 | 0.001 | — |
| sat haut | 0.308 | 0.285 | 0.263 | 0.249 | — |

Couple à 0.474 à l'entraînement, à confirmer au balayage (contrainte C2 : les
métriques d'entraînement surestiment d'un facteur ~10).

---

## 6. Ce qui reste ouvert

| | sujet | état |
|---|---|---|
| D1 | lever de pied | 0.0101 m pour 0.03–0.05 visés — **16 % de la cible** |
| D2 | déport arrière | **ne se reproduit pas en simulation** (talon 0.49–0.51). Vient du déploiement : filtre de la `PostureTask`, ou décalage de CoM réel. Se tranche sur les logs robot. |
| D3 | marche arrière | non mesurable en simu, elle y réussit |
| D5 | double appui | 0.109 pour 0.5 visé — **la politique ne respecte pas le rapport cyclique** de l'horloge, elle n'en garde que la lenteur |
| D6 | contacts propres | sans terme depuis le rejet de `cbal`. Reprise possible à 0.1 **avec porte sur la commande**. Décision de Léo. |
| C3 | limites de vitesse par articulation | à porter dans l'entraînement |
| — | asymétrie gauche/droite | poignet gauche saturé 0.365, droit 0.147, malgré la mirror loss. Non expliqué. |
| — | déploiement de l'horloge | 530 dims : `obs_format` neuf + horloge répliquée en C++ |

---

## 7. Erreurs de méthode à ne pas répéter

1. **Juger sur un point, pas sur une pente.** Le suivi remontait quand j'ai
   coupé la run 16.
2. **Ne pas conclure avant la fin du transitoire de reprise** (~500 itérations).
3. **Lire la config effective en entier**, pas le premier bloc qui donne raison :
   `RHPS1_SLOW_PERIOD` touchait le curriculum et pas la récompense, vingt lignes
   plus bas.
4. **Une variable d'environnement peut être lue par deux paliers.**
   `RHPS1_STEP_PERIOD` l'était déjà par `periodlive`.
5. **Vérifier qu'un patch atterrit au bon endroit**, pas seulement qu'il
   s'applique. Un motif présent deux fois se remplace une fois.
6. **Tester tout terme neuf sur un vrai `env.step()`** avant de lancer.
7. **Un moniteur qui ne se termine pas ne réveille personne.** Il journalise, il
   n'alerte pas.
8. **`pgrep -f` / `pkill -f` se trouvent eux-mêmes.** Passer les PID en argument.
9. **Le cwd du shell se réinitialise.** Chemins absolus.
10. **Le budget dit ce qu'un comportement coûte, pas s'il est atteignable
    depuis l'autre.** Un bassin dont la politique ne ressort pas ne se voit pas
    dans une somme de poids.


---

## 10. Bilan du 2026-09-03 et plan de test

### 10.1 Où on en est, objectif par objectif

| | objectif | état | mesure |
|---|---|---|---|
| O1 | couples faisables | **non tenu, et mal mesuré** | la demande dépasse la limite 2.35 % du temps, pic 7.6× |
| O2 | impact faible | tenu en simu, **non tenu en déploiement** | 0.1441 au balayage, ~150 N·m au genou en mc_mujoco |
| O3 | ne jamais tomber | tenu | 0.0000 |
| O4 | haut du corps calme | à juger sur vidéo | poignets saturés 0.38–0.53 à l'entraînement |
| D1 | lever de pied | tenu | 0.0326 pour 0.030 visé |
| D5 | cadence | quasi tenu | période 0.70–0.95 s contre 0.90 pour le BWC |
| D6 | contacts propres | partiel | pieds à plat OK, répartition de charge 0.25 |

### 10.2 La découverte qui invalide O1

**Aucune métrique du barème ne voit un dépassement de couple.** Les trois
qu'on surveillait mesurent l'aval du filet :

- `joint_torque_limit_margin_penalty` lit `data.actuator_force`, la force
  **appliquée** par MuJoCo, donc bornée par construction — son ratio ne peut
  pas dépasser 1.0 ;
- `torque_saturated_frac` compte la **fréquence** de l'écrêtage, pas son
  amplitude ;
- `TorqueRatio/*` sont des moyennes d'épisode et plafonnent à 0.69.

Or mjlab termine son PD par `torch.clamp(torque, ±force_limit)` et ce filet
n'existe nulle part en aval : mc_mujoco applique `MjRobot::PD` tel quel, le XML
déclare `forcelimited="false"`, et le vrai robot a des variateurs qui saturent.
L'entraînement rabote 18 % des pas sans jamais le facturer.

**Six critères sur six ont donc été validés sur une politique qui demande
jusqu'à 7.6 fois la limite.**

`torque_demand_overshoot` corrige la mesure : il lit
`FiniteDifferencePdActuator._raw_torque_peak`, le ratio que l'actionneur
calcule lui-même avant écrêtage. Le recalculer depuis `data.joint_vel_target`
donne un résultat faux — vérifié contre `actuator_force`, 50 N·m d'écart médian
— parce que l'actionneur emploie une vitesse filtrée par différence finie.

### 10.3 La réserve sur le couple subi est levée

Réserve initiale de Léo : à l'impact les 150 N·m seraient le couple **subi**, la
réaction du sol, et non le couple commandé — auquel cas un terme sur la demande
avant écrêtage ne le verrait pas.

**Vérifiée dans la source de mc_mujoco, et fausse.** `mj_sim.cpp:707` :

```cpp
torques[mj_jnt_to_rjo[i]] = data->qfrc_actuator[model->jnt_dofadr[mj_jnt_ids[i]]];
```

`qfrc_actuator` est la force des **actionneurs seuls**. La réaction du sol vit
dans `qfrc_constraint`, que mc_mujoco ne lit jamais. Et rien ne l'écrête :
`MjRobot::PD` retourne `kp*e_p + kd*e_v` nu, et la classe par défaut du XML
déclare `forcelimited="false" gear="1"`.

Les 150 N·m sont donc **le couple commandé avant écrêtage**. Le PD réagit à
l'impact — l'erreur de position explose, kp=20000 la multiplie — mais le pic est
produit par le régulateur, pas transmis par le sol.

Recoupement indépendant : la médiane des maxima de demande vaut 1.68 sur le
checkpoint déployé, soit 168 N·m pour une limite à 100. Léo mesure ~150. Les
deux mesures concordent, par deux chemins sans rapport.

**Conséquence : `torque_demand_overshoot` adresse directement ce que Léo
mesure.** L'instrumentation séparée du couple subi devient inutile.

### 10.4 Plan de test

**T1 — instrumenter. FAIT, sans entraînement.** La question qu'il posait
— la demande est-elle découplée du couple mesuré en déploiement ? — est tranchée
par la lecture de `mj_sim.cpp` en 10.3 : c'est la même grandeur. Et la demande
est déjà mesurée sur le checkpoint déployé : ratio max 7.62, médiane des maxima
1.68, 2.35 % des pas au-dessus.

**T2 — pénaliser la demande.** `torque_demand_overshoot` à poids réel, une seule
déviation depuis la configuration 6/6. Porte : `torque_demand_ratio_max` sous
2.0 et fraction au-dessus sous 0.005, sans perte sur D1 ni D5.

**T3 — annulé.** Reposait sur l'hypothèse d'un couple subi distinct de la
demande. 10.3 la réfute.

**T4 — référence BWC en entrée**, si T2 ne suffit pas. Les profils sont déjà
extraits : vol du pied par phase, déport latéral du CoM, durées d'appui,
placement au point de capture. En **observation**, pas en récompense, pour que
la politique garde le droit de s'en écarter.

**Ce que M1 a rapporté ici :** mesurer avant d'entraîner a supprimé deux des
quatre essais. T1 s'est résolu en lisant une ligne de source, et T3 n'avait pas
lieu d'être. Le coût en GPU de la règle est nul et son rendement est de deux
runs économisés.


### 10.5 Le depassement est dans les bras, pas dans le genou

Mesure par articulation sur le checkpoint 6/6, en conditions d'entrainement,
politique deterministe (`scripts/tools/size_demand_penalty.py`) :

```
joint          exces^2    part   ratioMax  fracOver
L_ELBOW_Y      0.52145   47.2%      12.53    0.0800
L_WRIST_R      0.43372   39.2%      11.00    0.0884
CHEST_P        0.04036    3.7%      12.15    0.1217
CHEST_Y        0.04004    3.6%       9.40    0.1214
...
R_KNEE_P       0.00000    0.0%       1.21    0.0001
L_KNEE_P       0.00000    0.0%       1.01    0.0000
L_CROTCH_P     0.00000    0.0%       0.92    0.0000
```

**Le genou ne depasse jamais** : 1.21x au pire, un joint-pas sur dix mille. Le
"mediane des maxima 1.68" de la veille etait un maximum sur TOUS les joints,
attribue au genou parce qu'il recoupait les 150 N.m. C'etait une coincidence.

Deux consequences.

**Une vraie faille O1 existe, mais dans le haut du corps.** Coude et poignet
demandent 12 fois leur limite et 86 % du terme vient d'eux. Sur le robot reel ce
sont les variateurs qui saturent. Asymetrie a expliquer : L_ELBOW_Y et L_WRIST_R
seuls, jamais leurs homologues droits, alors que le rung `mirror` est actif.

**Le terme ne traitera pas les 150 N.m du genou.** Sous QP, la sortie de la
politique devient une cible de PostureTask que le QP resout, et c'est SA sortie
qui entre dans le PD a kp=20000. 150 N.m y valent 0.0075 rad, soit **0.43
degre** d'erreur de suivi. Ce n'est pas une commande agressive, c'est un servo
tres raide qui encaisse un impact. Le BWC sous le meme kp ferait le meme pic si
son atterrissage etait aussi dur : il atterrit a 2.0x le poids, la politique a
3.0x.

**Le levier est donc la durete de l'impact**, et la divergence a expliquer est
deja mesuree : vitesse d'atterrissage **0.18 en mjlab contre 0.35 en
mc_mujoco**. Le facteur deux sur la vitesse est le facteur deux sur le couple.
Candidat principal : la PostureTask est un second ordre de constante 25 ms --
cinq pas de politique -- absente de l'entrainement, ou la politique commande le
PD directement. Un vol retarde ne decelere pas a temps avant la pose.

### 10.6 T2 lance, effondre, controle en cours

Reprise depuis 2026-09-01_17-45-07 model_4500, verifie sur un plateau avant
(C4) : fell_down 0.0000, sole_height_p90 0.0329->0.0368, mean_reward 78->84
monotones sur 4200-4800.

Premiere surprise des l'amorcage : a l'entrainement l'action est ECHANTILLONNEE,
et la demande explose par rapport a la politique deterministe.

```
                    deterministe    entrainement
fraction au-dessus         0.022            0.48
ratio max                   12.5             109
```

Le poids -0.04 avait ete dimensionne sur la colonne de gauche.

Puis l'effondrement, en treize iterations :

```
             4502      4504      4506      4508      4510      4512
reward    -6.7612  -10.3380  -51.8726  -72.3370  -71.0069  -81.7666
falls      0.0000    0.0000    5.5000   13.1250   10.6875    6.1667
Rdmd      -0.1862   -0.3429   -0.6799   -1.2267   -1.1907   -1.4223
```

La reprise part deja a -6.76 la ou la run d'origine etait a +80 a la meme
iteration, avec une penalite de -0.19. **La penalite moyenne n'explique pas la
chute** -- elle vaut 1.7 % de l'amplitude du reward.

Deux hypotheses, et elles donnent la meme courbe :

1. **C4** -- reprendre depuis ce checkpoint le casse, independamment du terme.
2. **La variance du terme** -- un exces au carre a ratio 121 vaut 14 400 sur un
   seul joint-pas. La moyenne reste petite parce que c'est rare ; le gradient
   sur cet echantillon est enorme, et PPO prend un pas destructeur.

Le controle -- meme reprise, sans le terme, 150 iterations -- est lance. Rien
d'autre ne peut les departager, et c'est exactement la regle "blamer le
checkpoint d'abord" qui exige de le lancer avant de corriger le terme.

Si le controle tient a +80 : le terme est coupable, et la correction est de
borner la contribution par articulation -- exces lineaire au lieu du carre, ce
qui transforme 121 en 120 plutot qu'en 14 400.

Si le controle s'effondre aussi : la methode T2 entiere est invalide, aucune
reprise depuis ce checkpoint n'est exploitable, et il faut repartir autrement.
