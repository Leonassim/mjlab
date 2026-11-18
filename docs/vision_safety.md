# Sécurité des politiques vision (rtx4080, 2026-11-18)

Réflexions et stratégie, discutées avec Léo, pour qu'une politique vision ne fasse jamais tomber le robot.
Elles valent pour les élèves image de `rtx4080` comme pour la politique à carte de `tardis`. Les résultats vivent
dans les journaux (`docs/coordination/*.md`) ; ce fichier garde la stratégie et ses raisons.

## Exigence

La vision ne doit jamais faire tomber le robot, quels que soient l'image, les coupures, les objets ou les gens dans
le champ (Léo, 2026-11-18). On teste sur le robot réel seulement quand c'est sûr dans nos bancs : la stabilité prime
sur la performance.

## Stratégie : trois couches

1. **La politique marche avec la vision**, entraînée avec un flux dégradé et la vraie caméra (pose mesurée par
   `tardis`, 15 à 17 Hz, latence de 30 à 150 ms, bruit, pixels perdus, courtes coupures).
2. **Un garde-fou simple dans le contrôleur** : si l'image est douteuse, il met la commande en pause. Le robot
   s'arrête gentiment, pieds posés, comme à commande nulle (pas d'arrêt d'urgence), puis repart quand l'image redevient
   normale depuis environ 1 s.
3. **La politique tient la transition** : entre le début du problème et l'arrêt, il se passe moins d'une seconde avec
   une image perturbée. On l'entraîne donc aussi avec ces perturbations (passants, coupures, image figée), le terrain
   restant inchangé : la bonne réponse est de ne pas réagir à ce qui n'est pas du sol.

**Pourquoi le garde-fou est dans le contrôleur, et pas dans la politique :** on peut le régler et le tester sans
réentraîner. Il est prévisible et explicable, alors qu'un réseau peut décider de continuer. Il sert à toutes les
politiques vision. Il réutilise un comportement déjà exigé et entraîné : la pause en pleine montée
(`PAUSE_P`, `STAIR_PAUSE_P`, `ARRIVE_LOADED` attend les deux pieds posés).

**La politique n'évite pas les obstacles.** Si quelqu'un bloque vraiment le chemin, c'est l'opérateur qui coupe la
commande. Ce qu'on garantit : une image perturbée ne fait jamais tomber le robot.

## Image « douteuse » : trois tests simples

Calculés sur chaque image reçue, sans réseau :

| Test | Critère (prototype) | Mesures en marche normale (banc dur, 128 envs) |
|---|---|---|
| Quelque chose devant | plus de 8 % des pixels à plus de 0,6 m au-dessus de la semelle la plus basse, ou plus de 15 % à moins de 0,6 m de la caméra | hauteur : 1,2 % en médiane, 3,4 % au pire ; un passant : 26 % en médiane. Distance : 0,7 % au pire |
| Image vide | moins de 10 % des pixels avec un retour | les pixels hors du cercle WFOV de 120° ne renvoient jamais rien : ne pas compter les pixels sans retour |
| Image figée ou en retard | horodatage de la dernière image plus vieux que 0,3 s | `vision_max_age` du contrôleur le fait déjà pour la carte |

- **La distance seule ne suffit pas.** Avec « à moins de 0,6 m », le passant n'a presque jamais déclenché la pause, et il
  a triplé les chutes. Le passant du banc est le plus souvent entre 0,6 et 1,5 m.
- **La hauteur des points discrimine.** Les points sont reconstruits avec la pose de la caméra et la profondeur. Une
  personne monte à plus d'un mètre, une marche du curriculum jamais au-dessus de quelques dizaines de centimètres
  devant le robot.
- **Il faut la pose caméra au moment de l'image.** Sur le robot, elle vient de la cinématique et de l'horodatage, comme
  pour la carte.
- **Limite connue : un objet bas et inconnu plus loin** n'est pas attrapé, par exemple la boîte de 25 cm à 0,8 m du
  slot 23. Deux pistes :
  - des blocs fantômes à l'entraînement, ce que fait déjà vis3 ;
  - comparer l'image à la reconstruction du scan par l'élève (sa tête `--scan-aux`) et s'arrêter si elle ne ressemble
    à aucune marche connue.

## Banc de robustesse (`scripts/stairs/depth_eval.py`)

Mode par défaut (protocole vismoke). Les pannes sont injectées dans l'image lue par l'élève, toutes les `EV_T` s
pendant `EV_D` s (8 et 3 par défaut).

| Variable | Effet |
|---|---|
| `SCEN=out` | plus aucun retour |
| `SCEN=freeze` | la dernière image tenue |
| `SCEN=person` | une bande verticale proche (0,4 à 1,5 m) qui traverse l'image |
| `GATE=1` | prototype du garde-fou (pause de la commande d'escalier) |
| `GATE_NEAR`, `GATE_FRAC`, `GATE_HIGH`, `GATE_HFRAC`, `GATE_VALID`, `GATE_FROZEN`, `GATE_HOLD` | seuils du garde-fou |

- **La sortie compte à part les chutes pendant une panne et dans la seconde qui suit.** Le banc dur a une base
  d'environ 50 chutes, due aux marches difficiles.
- **Bruit du banc : ±10 chutes** entre deux tirages GPU.
- **Couper les coupures aléatoires de l'entraînement** pour ces bancs (`RHPS1_DEPTH_BLACK_P=0`), sinon elles brouillent
  la mesure.

**Critère pour un candidat robot :** 0 chute attribuable à la vision dans tous les scénarios, avec le garde-fou, et pas
de pas parasite à commande nulle. S'y ajoutent les bancs habituels (marche, montée, descente, arrêt en plein
escalier) et, dans mc_mujoco, le banc obligatoire de Léo : boîte de 25 cm à commande nulle, carte coupée et carte
figée.

## Premiers résultats (rl2 1000, ancienne pose caméra, 128 envs × 24 s)

| Panne | Sans garde-fou | Avec garde-fou (distance seule) |
|---|---|---|
| aucune | 54 arrivées / 49 chutes | 49 / 61 (jamais déclenché) |
| passant | 21 / 168 | 25 / 159 (presque jamais déclenché) |
| image coupée | 53 / 55 | 26 / 36 |
| image figée | 56 / 51 | 27 / 45 |

Le passant est la panne la plus grave. La suite figure dans le journal `rtx4080`.

## Ce qui reste à décider

- **Implémenter le garde-fou dans rl_controller** : à faire avec Léo, qui doit l'approuver. Il doit passer par la même
  pause que la politique a apprise.
- **Vérifier qu'un arrêt n'importe où dans l'escalier est sûr**, un pied sur une marche.
- **Choisir l'équivalent pour la politique à carte** : le garde-fou ou la carte lue comme plate. Une carte plate sur un
  escalier réel est dangereuse en descente ; la pause est probablement plus sûre.
