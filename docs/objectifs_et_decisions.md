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
