# Repartir de policy 0, avec un curriculum

Plan pour un run unique et long depuis la configuration de policy 0, portant
tout ce que la campagne du 24-25 août a appris. Il remplace la méthode qui a
échoué : sept runs courts, sept points de départ, des seuils déplacés en cours
de route, et des verdicts rendus sur des métriques d'entraînement qui mentent.

## Critères d'acceptation

Ce sont ceux de Léo, dans son ordre de priorité. Ils portent sur une **plage de
commandes**, pas sur une moyenne.

| | seuil | mesuré sur |
|---|---|---|
| ne jamais tomber | 0 chute sur la grille de commandes | balayage déterministe |
| couples faisables | écrêtage jambes < 0.25 | balayage |
| impact faible | vitesse pré-contact < 0.16 | balayage |
| pieds à plat | 4 coins au contact | balayage |
| lever de pied | garde au sol réelle >= 0.030 | balayage |
| immobile à l'arrêt | pas de dérive commande nulle | balayage |

Référence : policy 0 tient les cinq premiers et échoue sur le lever de pied.
it15600 tient le lever et échoue sur l'impact et les chutes.

## Phase 0 — réparer l'évaluation avant de lancer quoi que ce soit

**C'est le préalable, pas une option.** L'entraînement a annoncé 3.3 % de chutes
sur une policy qui tombait en mc_mujoco au-dessus de 0.16 m/s. Tant que
l'évaluation ment, chaque décision est du bruit.

`scripts/tools/sweep_eval.py` : pour un checkpoint donné, dérouler la grille
(vx ∈ {0, 0.1, 0.2, 0.3}, vy ∈ {0, ±0.2}, yaw ∈ {0, ±0.3}), N secondes par
point, sortir chutes / garde au sol / impact / écrêtage jambes / contacts plats
**par commande**. Sortie : un tableau, pas une moyenne.

Un checkpoint est accepté ou rejeté là-dessus. Jamais sur les courbes
d'entraînement.

## Phase 1 — les corrections qui ne changent pas le comportement

À appliquer avant le lancement, parce qu'elles rendent le reste observable :

- **`soleclear`** — la garde au sol mesurée au point le plus bas de la semelle,
  débord de boîte compris, et non à un site au milieu. Sans ça, tout chiffre de
  hauteur de pied est gonflé par l'inclinaison : mesuré 4.8 cm là où le vrai
  dégagement était 2.0.
- **écrêtage jambes séparé de l'agrégé** — deux tiers de l'agrégé sont poignets
  et épaules, et le couple n'est pas partagé entre articulations.
- **`flat_support` normalisé par appui**, et non par seconde. Il mesure une
  propriété par événement facturée au temps, donc atterrir moins longtemps est
  la façon la moins chère de moins payer un mauvais atterrissage. C'est ce qui
  a tenu la période à 0.45 sous cinq pondérations différentes. La version
  bâclée (poids divisé par deux) a servi de test ; ici on le fait proprement.
- **`posture_stiffness`** fonctionne depuis le 24 août (surcharge non-const
  supprimée). Tous les slots déclarent 8000, la valeur réellement exécutée.

## Phase 2 — le curriculum

Six étapes. **Chacune avance sur une réussite mesurée, jamais sur un compteur
de pas** — le défaut documenté des échelles précédentes, où les paliers
défilaient pendant que la grandeur visée ne bougeait pas.

| étape | ce qu'on ajoute | condition pour passer |
|---|---|---|
| S0 | config policy 0 sur le code d'aujourd'hui | chutes < 1 %, marche confirmée |
| S1 | `soleclear`, cible juste au-dessus du mesuré | garde >= 0.020, chutes < 2 % |
| S2 | échelle de foulée | foulée >= 0.060 |
| S3 | `landtime` + période | période >= 0.45, chutes < 2 % |
| S4 | garde au sol poussée à 0.030 | garde >= 0.030 |
| S5 | plage de commandes élargie | balayage sans chute jusqu'à 0.25 m/s |

S0 n'est pas une formalité : il vérifie que les changements de code du 24-25
n'ont pas cassé la policy qui marchait.

## Phase 3 — garde-fous, figés au lancement

Sur les critères de déploiement uniquement, jamais sur l'objectif. Relatifs à
la ligne de base du run, avec un CAP absolu qu'aucune ligne de base ne déplace,
et **posés au-dessus des cibles que les récompenses visent** — un plafond posé
sur la cible se déclenche quand la récompense fait son travail.

Ils ne bougent plus une fois le run lancé. Les déplacer en cours de route, ce
qui est arrivé quatre fois, rend tout verdict ininterprétable.

## Ce qu'on sait déjà qui va coincer

**Le lacet de hanche.** `CROTCH_Y` tourne à 0.82 de sa limite de couple contre
0.60 pour le genou, et `error_vel_yaw` vaut 0.286 : le robot tourne alors qu'on
ne le lui demande pas, et c'est ce lacet parasite qui charge la hanche. La
foulée a buté dessus, pas sur la puissance du genou.

Libérer le balancement des bras n'a rien changé en 450 itérations — retirer une
pénalité donne la permission, pas le comportement. **Il faut chercher d'où vient
le lacet** (asymétrie de la démarche, pivot du pied en appui) plutôt que
d'essayer de le compenser.

À mesurer dès S0, sur policy 0 : si le lacet parasite est déjà là avec des pas
de 3 cm, il est structurel et il plafonnera tout. S'il apparaît avec la foulée,
il est une conséquence et se traite avec elle.

**Les objectifs se disputent le budget de couple.** vitesse = foulée / période :
allonger la période réduit la vitesse à foulée constante, et lever plus haut
coûte plus d'impact à l'atterrissage. Les quatre ne tiendront pas ensemble à
0.3 m/s. S5 dira où se situe la limite réelle, plutôt que de la supposer.
