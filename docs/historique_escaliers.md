# Historique du chantier escaliers RHPS1 (6 – 15 novembre 2026)

Ce document retrace les recherches, les essais et les raisonnements du sujet « monter et descendre des
escaliers » avec RHPS1 en apprentissage par renforcement : ce qui a été tenté, pourquoi, ce que ça a donné et ce
qu'on en a retenu. Les scripts de chaque essai sont dans `scripts/stairs/history/`, les checkpoints de
référence dans `checkpoints/rhps1/`, la marche à suivre pour reproduire dans `scripts/stairs/README.md`.

## 1. Objectif et cadre

**Le but (Léo, 6 novembre).** Monter et descendre 3-4 marches d'un escalier de bâtiment japonais (contremarche
≤ 20 cm, giron ≥ 24 cm ; typiquement 16-18 / 26-30 cm), avec la vision dans la boucle (Azure Kinect sur la tête).
Seuil fixé le 13 novembre : « si ça marche déjà très bien pour 5 cm, ce sera déjà super ».

**Les exigences de comportement**, accumulées au fil des retours :
- lent et contrôlé, sans imposer de vitesse ; impacts doux (vitesse de pose du pied au niveau du BWC, ~0,2 m/s) ;
- tronc neutre, pieds à leur écartement initial, pas d'appui prolongé sur une jambe ;
- s'arrêter en haut sur les deux pieds ; repartir du haut pour descendre ; quand on relâche le stick sur une
  marche, s'arrêter sans tomber (deux pieds sur la même marche de préférence) ;
- marcher naturellement sur le plat (« pour pas qu'il ait l'air débile ») et rester stable à l'arrêt ;
- **ne pas compter sur la saturation** : couple max respecté par articulation, genou à 108 N·m partout ;
- **RL complet, sans imitation ni robot fantôme** (décision du 9 novembre).

**Le cadre technique.** mjlab (fork `Leonassim/mjlab`), tâche dérivée du tracking RHPS1 (`stairs.py`), terrain
pyramide de 3 marches (palier 1,6 m, giron 0,27/0,30 m), carte de hauteur 1,2 × 0,6 m au pas de 5 cm alignée sur
le cap (325 entrées, encodeur 128-32), plante complète du robot : drivers des vérins, retard bas niveau, retard
d'actionneur, randomisations (friction 0,3-1,2, biais, gains, centre de masse). Déploiement via rl_controller
(slot 22, `obs_format 11`, scan calculé analytiquement dans mc_mujoco).

## 2. Mise en place (6-7 novembre)

- **Faisabilité statique** : avec le genou à 100 N·m, 18 cm est faisable avec 20-35° d'inclinaison du tronc ;
  la cheville en tangage (65 N·m) et le lacet de hanche (35 N·m) deviennent ensuite limitants. Le genou de
  l'appui avant travaille à 55-80°, contre ~30° en marche.
- **Plan vision** (validé) : un professeur entraîné avec la carte exacte, puis un élève distillé sur une carte
  dégradée produite par le PC vision (Azure + odométrie) dans le même format.
- **Pièges trouvés en montant la tâche** : importer `rl_ext` réécrivait l'échelle d'action partagée (genou
  0,0525 → 0,081) ; les dalles plates en boîte faisaient tomber le robot (contact boîte-boîte, corrigé avec le
  collisionneur natif) ; 4096 envs saturent les 11 Go de la 2080 Ti (3584 retenus).

## 3. Phase 1 — copie BWC sur le plat, RL sur les escaliers (st1-st22, 7-8 novembre)

Idée : garder la marche du slot 20 par imitation sur les dalles plates et apprendre les escaliers par
récompenses propres.

- **st1-st8** : effondrements et dérives. Le slot 20 aveugle trébuche dès 2 cm ; les termes d'équilibre
  (bonus statique, point de capture) ont été contournés en allongeant l'appui simple (86 % du temps).
- **Découverte clé (st11)** : avec le bruit d'exploration, les régularisations de la copie mangeaient tout le
  bonus de survie. Avec un horizon de 0,5 s, une chute ne coûtait presque rien sur les escaliers. Leçon :
  **toujours mesurer le taux net de récompense stochastique d'un nouveau type d'env avant d'entraîner.**
- **st14** : le noyau de vitesse payait davantage l'immobilité que la marche (l'oscillation de vitesse
  intra-pas rend l'erreur instantanée de la marche aussi grande que la cible). Remplacé par une récompense de
  progression sur vitesse filtrée.
- **st15-st20** : un ancrage quadratique sur les actions du slot 20 a tenu, puis a explosé dès que la tâche
  demandait une démarche différente (st19, st20) : une ancre non bornée est une impasse.
- **st17** : passage à γ 0,9975 / λ 0,987, soit l'horizon de 2 s de l'état de l'art à 50 Hz.
- **st21-st22** : l'expert escaliers seul monte un peu mais ne tient jamais une arrivée au-dessus de 0,5 cm ;
  « marcher et voir si ça tombe », pas de vraie stratégie de pied posé.

## 4. Phase 2 — RL complet à 200 Hz (rl1-rl23, 9-11 novembre)

Décision de Léo le 9 novembre : arrêter l'imitation, RL complet façon état de l'art, à son rythme, lentement.

- **rl1-rl2** : depuis le slot 20, parcours à 0,3 m/s sans s'arrêter ; plafond de progression ajouté.
- **rl3-rl7, depuis zéro** : les épisodes raccourcissaient, parce que **vivre coûtait plus cher que tomber**
  (-36/s de régularisations sous un bruit fort, contre une chute à -30). La chute passe à -200 (rl7) : survie et
  premières arrivées tenues.
- **rl8-rl10** : impacts 4-8 fois le poids, puis coûts d'atterrissage linéaires et plafonnés, tronc, vitesse
  verticale, couple. À pleine charge des régularisations, le robot fige (rl9) ; progression ×2 (rl10).
- **rl11-rl15** : dégagement des pieds, récompense par marche franchie, curriculum plus fin (20 niveaux),
  arrivée validée avec les deux pieds passés.
- **rl16-rl23, nouveaux départs de zéro** avec tout le jeu de récompenses : encodeur de scan, scan décalé de
  0,3 m vers l'avant, chute rampée de -30 à -200, cadence lente à double sens (debout ne rapporte rien, maximum à
  un pas toutes les 2 s). C'est l'origine de la lignée suivante.

## 5. Phase 3 — la lignée RL et le slot 22 (rl24-rl48b, 11-13 novembre)

- **rl24-rl37 — l'appui sur une jambe** : le robot restait des secondes sur une jambe (avant les marches, après
  l'arrivée). Traité par un plafond de balancier à 1,2 s, puis une terminaison si un pied reste levé plus de 2 s,
  un pied « posé » seulement à 15 % du poids pendant 0,1 s, la récompense d'arrêt payée sur deux pieds, des
  pauses de commande en marche, le bonus d'arrivée une fois stabilisé 0,5 s, et le bruit réduit (rl34).
- **rl38-rl40 — les descentes** : abandonnées car un atterrissage dur coûtait plus que rester en haut ;
  plafond des coûts d'atterrissage, immobilité plus chère, départs sur une marche intermédiaire.
- **rl41-rl43** : mode fixe par env (curricula montée et descente séparés), rejeu de niveaux, enveloppe de
  marche du slot 18 (vitesses articulaires, cheville, pied en l'air), limites de course des vérins.
- **rl40-rl46 — le taux d'apprentissage gelé** : les moyennes d'action partaient bien au-delà de l'écrêtage
  (±12, jusqu'à 200). Il n'y avait plus de gradient, mais la KL explosait, ce qui collait le taux d'apprentissage
  au plancher 1e-5 : tous les « plateaux » de rl40-43 étaient du bruit autour d'une politique figée. Corrigé par
  une pénalité sur les moyennes au-delà de 10 et une KL calculée sur les actions écrêtées (rl45-46). Leçon :
  **quand un run stagne, regarder `Loss/learning_rate` d'abord.**
- **rl47-rl48b** : descentes depuis le palier, niveaux sauvegardés entre reprises, coût d'inclinaison du pied.
  **rl48b model_31500 = slot 22** : premier à monter 2 cm puis 5 cm dans mc_mujoco à l'écran (environ une fois
  sur deux), une fois corrigé le plafond du genou du patch mc_mujoco (70 → 108 N·m, valeur codée en dur).

## 6. Phase 4 — couples et genou (rl49-rl54, 13-14 novembre)

- **rl49** : vitesse plafonnée à 0,15 m/s. **rl50** : coût de couple par articulation au-delà de 90 % de la
  limite. **rl51** : terminaison si une articulation de jambe reste à ≥ 97 % de sa limite plus de 0,4 s
  (contrainte comme terminaison).
- **Découverte du 14 novembre** : à toutes les hauteurs, même 0,5-1,4 cm, la politique faisait des fentes de
  ~35 cm (le double du BWC) et se réceptionnait sur un genou à 55-60°. Le genou portait alors le corps à 108 N·m
  pendant 0,5-1 s, puis cédait. Un coût de couple ne peut rien contre ça : saturé, le couple ne varie plus avec
  la flexion. **rl52-53** : coût direct sur la flexion du genou chargé au-delà de 40° + 2°/cm ; au poids -100,
  les genoux au-delà de 45° passent de 20 % à 2-5 % du temps d'appui.
- **rl54** : curriculum plafonné à 6 cm, pour concentrer l'entraînement sur l'objectif 5 cm.
- **Calibrage des bancs** :
  - Le banc de chutes mjlab est très sévère : le slot 20, qui marche sur le vrai robot, y tombe 23 fois sur 32,
    et un écart de moins de ~7 chutes sur 48 est du bruit. **Le juge principal devient mc_mujoco** (12 essais).
  - La perte du critique à 40-80 n'était pas une panne : il expliquait 90 % de la variance, la valeur venait de
    l'échelle des récompenses (chute -200).
- **Bilan** : aucun checkpoint fiable. En mc_mujoco, rl51 37000 atteint le palier 8 fois sur 12 mais tombe
  ensuite 10 fois sur 12, et rl54 42500 atteint le palier 5 fois sur 12 (4 sans chute). Le diagnostic retenu :
  la marche de base de la lignée s'était dégradée (petits pas traînés, fentes, genoux pliés), parce que
  l'entraînement ne contenait que des escaliers.

## 7. Phase 5 — la tâche d'approche (ap1-ap7, 14 novembre)

Demande de Léo : partir du plat, à distance et en angle, pour placer les pieds avant d'attaquer ; s'arrêter sur
les marches quand on relâche le stick.

- **ap1 → ap2** : redémarrage depuis rl54 plutôt que depuis le slot 22 (mesuré : slot 22 = 2 paliers sur 12).
- **ap2-ap3 — figés** : la commande « tourner + avancer » (jamais vue sur les escaliers) faisait figer la
  politique, puis le blocage s'est généralisé. Retirée (ap4) ; l'approche s'ouvre progressivement (de 0,15-0,35 m
  à 0,5-1,5 m, de ±9° à ±30°).
- **ap5 — figé aussi** : renforcer la récompense d'alignement payait le fait de rester immobile bien aligné.
  Leçon : **pour une posture tenue aussi à l'arrêt, mettre un coût, pas une récompense** (ap6 : pénalité
  au-delà de 10° d'écart de cap).
- **ap6-ap7** : l'entraînement progressait, mais la politique déterministe restait timide (0,02-0,06 m/s). Le
  bruit d'exploration faisait marcher le robot en entraînement.
- **Fusion du mjlab principal** (15 novembre) : MuJoCo 4.0.1, torch 2.14. Corrections importantes : le retard
  des actionneurs était réécrit pour tous les envs à chaque réinitialisation partielle, d'où un retard effectif
  d'environ 2/3 du réglage, et des capteurs pouvaient être lus en retard. La physique d'une même politique est
  restée équivalente (vérifié) et les itérations sont passées de 5 s à 4,3 s.

## 8. Phase 6 — 50 Hz depuis zéro (s50, s50e, 14-15 novembre)

Idée de Léo : à 50 Hz, comme l'état de l'art. La question de la rigidité du robot a été discutée : les phénomènes
rapides (impacts, asservissement) restent dans la boucle bas niveau à 1 kHz. La politique ne peut agir qu'à
travers la réponse des articulations (4-8 Hz mesurés), donc même à 200 Hz elle ne corrige pas pendant l'impact.
Le vrai coût du 50 Hz est la latence de décision (jusqu'à 20 ms).

**Adaptations vérifiées** (robot debout aux deux fréquences : même coût par seconde à ±15 %) :
- l'action rate est le seul terme qui compare deux pas : poids ÷ 16 (le même mouvement coûte 16 fois plus à
  50 Hz ; mesuré -0,05/s pour le mouvement contre -0,46/s pour le bruit) ;
- fenêtre de vitesse de pose du pied en durée (50 ms) ; pic d'impact pris sur les 8 sous-pas de physique ;
- γ 0,99 / λ 0,95 (mêmes horizons en secondes) ; vidéos de 150 pas (3 s) ;
- la référence BWC joue 4 fois plus lentement, sans effet : rien ne s'en sert en RL ;
- côté contrôleur, `policy_step_size: 0.02` suffit pour le slot 22 (consigne maintenue entre deux décisions) ;
  surcharge de banc `RLQP_BENCH_POLICY_STEP` ajoutée pour mc_mujoco.

**Tâche s50** : 80 % de pyramides, 20 % de plat en RL, 10 % d'épisodes sans commande (debout), approche de 0,3 à
2 m à ±30°, arrêts au hasard sur l'escalier, un peu de plat après la descente, régularisations et contraintes
rampées sur 3000 itérations, chutes peu coûteuses au début, curriculum de 0,5 cm plafonné à 6 cm.

**Résultats** :
- En 250 itérations (1 h), les épisodes passent de 2,7 à 17,8 s et les arrivées tenues de 0 à 28 % ; perte du
  critique ~7 au lieu de 40-80.
- s50 à 3000 : déterministe bien meilleur (29 arrivées sur 32, plus de blocage), mais 0 palier en mc_mujoco, et
  l'écart-type de la politique est monté de 1,0 à 1,75. **Le bonus d'entropie 0,002 rendait la politique de plus
  en plus bruitée** (saturations ×7, appuis sur une jambe ×6).
- **s50e** (entropie 0) **à 3500 : meilleur résultat à ce jour en mc_mujoco**. Palier atteint 7 fois sur 12,
  sans chute 10 fois sur 12, deux traversées complètes avec descente de l'autre côté. **À 5 cm : palier 4/4**
  (une chute après 2,7 m) ; à 6 cm, arrêt sans tentative ni chute ; genou ≥ 105 N·m seulement 5 % du temps.

- **s50e à 10500 (16 novembre, après un week-end d'entraînement)** : palier 12 fois sur 12, descente complète
  de l'autre côté 12 fois sur 12, sans chute 11 fois sur 12, à 4, 5 et 6 cm. La seule chute survient à 5 cm
  après 4,2 m. Genou ≥ 105 N·m 2,7-7,9 % du temps. Stable depuis 7000 itérations (10-12 paliers et 8-11 sans
  chute sur 12 à chaque contrôle). Curriculum ~4 cm en moyenne sur 0,5-6 cm, arrivées tenues 52 %, écart-type
  encore 1,43 et en baisse lente.

Coût : 13,8 s par itération à 50 Hz contre 4,3 s à 200 Hz, mais chaque itération couvre 4 fois plus de temps
simulé, soit ~24 % de marche simulée en plus par heure de calcul.

## 9. Leçons transversales

1. **Mesurer en déterministe, pas sur les courbes d'entraînement** : le bruit d'exploration peut faire marcher
   en entraînement une politique qui fige une fois déployée.
2. **Le juge est mc_mujoco**, à la bonne fréquence. Le banc mjlab est sévère et bruité ; les métriques
   d'entraînement mélangent bruit et curriculum.
3. **Rester immobile est la réponse par défaut au risque** : chaque fois qu'avancer devient risqué ou que rester
   aligné ou immobile est payé, la politique fige. Il faut vérifier le bilan « marcher contre rester » avant de
   lancer.
4. **Vivre ne doit pas coûter plus que tomber** : coût des régularisations sous bruit × horizon < coût d'une chute.
5. **Coûts plutôt que récompenses** pour les postures à tenir aussi à l'arrêt ; **pas d'ancre non bornée**.
6. **Un couple saturé n'a pas de gradient** : contraindre la cause (flexion du genou), pas le couple.
7. **Quand un run stagne, regarder le taux d'apprentissage** (le plancher de la KL a gelé rl40-43).
8. **Une seule modification à la fois**, sinon un run entier devient impossible à attribuer.
9. **Les rampes repartent à zéro à la reprise** si on ne fixe pas leur décalage (ramp offsets).
10. **Entropie** : 0,002 en départ de zéro fait monter le bruit ; passer à 0 après quelques milliers d'itérations.

## 10. État au 15 novembre et pistes

- **En cours** : s50e (50 Hz), juge mc_mujoco toutes les 500 itérations. Meilleur checkpoint :
  `checkpoints/rhps1/s50e_50hz_10500` (12/12 paliers et descentes, 11/12 sans chute en mc_mujoco).
- **Pour le vrai robot** : créer un slot 50 Hz (`policy_step_size: 0.02`) dans rl_controller, vérifier les
  limites (vitesses articulaires, couples, impacts) sur les logs mc_mujoco, comme pour les slots précédents.
- **Pistes** :
  - monter le plafond du curriculum au-delà de 6 cm une fois 5 cm fiable ;
  - qualité de la marche à plat et stabilité à l'arrêt, mesurées sur les envs plats et debout ;
  - descente depuis le haut ;
  - distillation vers un élève sur carte dégradée (vision).
