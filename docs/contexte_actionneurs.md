# Contexte — modes d'actionnement, sim vs robot réel

Fichier de passation. À lire en entier avant de proposer quoi que ce soit :
la question posée est en grande partie déjà instruite, et la réponse naïve
(« écrivons un actionneur Elmo ») a déjà été écrite et délibérément **pas**
activée.

## La question

Peut-on avoir dans MuJoCo un actionneur plus fidèle que le PD actuel, sachant
qu'on a de quoi reconstruire la boucle **PPI en position** des variateurs du
vrai robot ?

## Réponse courte : c'est écrit, pas branché

`src/mjlab/actuator/elmo_replica_actuator.py` et
`elmo_replica_differential_actuator.py` implémentent déjà la cascade réelle :

```
e_pos   = q_ref_count - q_count
v_ref   = Kp_pos * e_pos + v_ff_count
e_vel   = v_ref - qdot_count
i_cmd   = Kp_vel * e_vel + Ki_vel * ∫e_vel      saturé à ±i_limit, anti-windup
tau     = eta * N * Kt * i_cmd
```

(`elmo_replica_actuator.py:183-213`, `COUNTS_PER_REV = 65536`.)

Les configs sont peuplées avec les **vraies** valeurs par articulation
(`asset_zoo/robots/RHPS1/rhps1_constants.py:558+`), lues dans
`Leonassim/RHPS1_gains` → `FromRealRobot/drive_gains_map.csv` :
`Kp_pos←KP3`, `Kp_vel←KP2`, `Ki_vel←KI2`, `gear_ratio←gear_ratio_N`,
`torque_constant←torque_constant_Nm_per_Arms`, limites de courant continu/pic.

Elles ne sont **pas** dans `RHPS1_ACTUATORS` (`rhps1_constants.py:539`), qui
reste en `FiniteDifferencePdActuatorCfg`. Le commentaire du bloc dit pourquoi :
« swap them in (and re-tune/re-train) deliberately, not as a drive-by change ».

## Les trois chaînes d'actionnement

| | mjlab (entraînement) | mc_mujoco (déploiement simulé) | robot réel |
|---|---|---|---|
| intégrateur | `implicitfast` | **Euler** | — |
| pas | 0.0025 | 0.001 | 0.005 (policy) |
| armature | valeurs réelles | **1.0** (placeholder XML) | — |
| boucle basse | PD sur cible position | PD en C++ | **cascade PPI dans le variateur Elmo** |
| couche QP | aucune | mc_rtc PostureTask + QP | idem |

- mjlab : `FiniteDifferencePdActuator`, vitesse désirée par différence finie sur
  la cible, filtre EMA `velocity_target_filter_alpha = 0.8`, projection de
  faisabilité couple `torque_feasibility_ratio`.
- mc_mujoco : `mc_mujoco/src/mj_sim.cpp:862`,
  `PD(rjo_id, q_ref, encoders, pd_zero_ref_vel ? 0.0 : alpha_ref, alphas)`.
  Le `/ratio` juste après est neutre — MuJoCo remultiplie par `actuator_gear`.
- robot : `~/src/rhps1-iob` (`RHPS1.conf`, `robot.cpp`) parle aux variateurs.

## Ce qui est déjà tranché — ne pas reprendre à zéro

**Le `gear` MuJoCo n'a rien à voir avec l'armature.** MuJoCo ajoute `armature`
brut à la diagonale de la matrice de masse, sans jamais la multiplier par
`gear`. Le N² est déjà dans la formule de calibration
(`armature = n_channels × JM × N²`). Ne pas y toucher.

**La PostureTask mc_rtc** est un second ordre :
`alphaD = K(q* − q) + D(refVel − alpha) + refAccel`, `D = 2√K`
(vérifié dans `~/src/Tasks/src/QPTasks.cpp`). Préserver la marge `√K·dt = 0.2`
donne `K = 0.04/dt²` : 1600 à 5 ms, 40000 à 1 ms. Le 1600 empirique de Léo sur
robot est exactement la marge de la simulation, pas un réglage au hasard.

**Le filtre PostureTask a été essayé côté entraînement et retiré**
(`finite_difference_pd_actuator.py:28,261` ; `_POSTURE_TASK_STIFFNESS = None`
à `rhps1_constants.py:961`). Deux raisons, toutes deux dans le commentaire :
l'entraînement le plaçait **en amont** de la différence finie et de la
projection, alors que la PostureTask mc_rtc consomme la cible, donc **en aval**
des deux — ordre irreproductible par construction ; et la politique produite
était instable dans mc_mujoco même debout. Le retard modélisé est réel : s'il
revient, il doit revenir là où le chemin de déploiement peut le placer.

**Les constantes de couple ont été corrigées le 2026-08-21** (elles étaient 2 à
5× trop basses) : `CROTCH_Y 0.116`, `KNEE_P 0.424`, `SHOULDER_P`/`CHEST 0.246`.
**`HEAD`, `SHOULDER_R/Y`, `ELBOW`, `WRIST` portent encore l'ancienne famille non
vérifiée (0.0458/0.0487)** — à ne pas croire tant que ce n'est pas sourcé. La
limite de 70 N·m au genou est le réducteur, pas le moteur.

**La projection de faisabilité couple est du *plant*, pas un garde-fou.** À
ratio 1.0 elle délivre exactement le couple que l'écrêtage d'effort de MuJoCo
délivrait déjà (démonstration dans le docstring, `finite_difference_pd_actuator.py`).
Une politique à ratio 1.0 dépasse sa fenêtre d'environ 10× par construction.

## Limites connues de la réplique Elmo — les lire avant de l'activer

Toutes listées en tête de `elmo_replica_actuator.py` :

1. **8 articulations non couvertes** : hanche roulis/tangage et cheville
   roulis/tangage sont des vérins linéaires en parallèle. Les limites de courant
   sont connues, mais convertir l'effort côté variateur en couple articulaire
   demande la géométrie des points d'attache des vérins, **absente de tous les
   fichiers accessibles**. Seules les 22 articulations `ACTUATOR_TYPE_ROTATE`
   sont modélisées. C'est le blocage principal : les jambes marchent
   essentiellement sur les 8 manquantes.
2. **`eta = 1.0`** partout (borne haute). L'estimation calibrée ~0.77 est
   spécifique au genou et non recoupée ailleurs.
3. **`i_limit` est une constante** au régime continu `CL[1]`. Le vrai variateur
   tolère un pic `PL[1]` pendant `PL[2]` (~8 s) avant de dérater. Dynamique à
   deux étages **non modélisée**.
4. **Pas de dérating back-EMF** : `tau_max` indépendant de `qdot`.
5. **Anti-windup supposé**, hypothèse d'asservissement industriel standard, non
   vérifiée dans la doc Elmo Gold-line de ce variateur.

## Contradiction à lever

`mj_sim.cpp:855-859` affirme que le robot ne reçoit **qu'une position** et que
son PD embarqué n'a pas de référence de vitesse. La réplique Elmo a bien un
`v_ff_count`, mais nul par défaut, ce qui est cohérent
(`elmo_replica_actuator.py:191-195`, « matching the real robot's current
FF-disabled state »). À confirmer côté `RHPS1_gains/README.md` section
« Feedforward and Sampling » avant de s'appuyer dessus.

## Où sont les infos

| quoi | où |
|---|---|
| actionneur d'entraînement | `src/mjlab/actuator/finite_difference_pd_actuator.py` |
| réplique Elmo | `src/mjlab/actuator/elmo_replica_actuator.py` (+ `_differential_`) |
| configs par articulation | `src/mjlab/asset_zoo/robots/RHPS1/rhps1_constants.py` |
| PD de mc_mujoco | `~/src/mc_mujoco/src/mj_sim.cpp:820-875` |
| PostureTask mc_rtc | `~/src/Tasks/src/QPTasks.cpp` |
| contrôleur déployé | `~/src/rl_controller/src/NewRLQPController.cpp` |
| interface robot réel | `~/src/rhps1-iob/` |
| description MuJoCo | `~/src/rhps1_mj_description/xml/RHPS1main.xml` |
| armature réelle | `~/mc-rtc-superbuild/etc/launchers/rhps1_real_armature.py`, variable `RHPS1_ARMATURE_ONLY` (commit `6d12a77`) |
| plan armature/posture | `~/.claude/plans/rustling-dreaming-piglet.md` |

**Pas sur cette machine** : le dépôt `Leonassim/RHPS1_gains`, qui contient
`FromRealRobot/drive_gains_map.csv`, `FromRealRobot/joint_torque_limits_rotate.csv`,
le `README.md` avec la table « Actuator Limits », `pdgains/elmo_joint_controller.py`,
`rl_controller/src/ElmoJointReplica.h` et le notebook (`simulate_closed_loop`,
calibration d'eta). Le cloner si les valeurs doivent être revérifiées.

Le `scripts/ppc/diag_vibration.py` cité par le plan armature est introuvable
sur cette machine.

## Question ouverte adjacente : l'armature

Avec les vraies armatures, mc_mujoco vibre violemment et la politique ne marche
plus ; à 1.0 elle remarche. L'entraînement, lui, tourne avec les vraies valeurs
sans problème. Écarté : valeurs identiques des deux côtés, XML dérivé propre,
fréquence propre par articulation sous le seuil d'instabilité à 1 kHz (pire cas
poignet `ωdt = 0.97` contre un seuil de 2) — donc pas une résonance mono-axe.
Le suspect restant est l'**intégrateur Euler de mc_mujoco** contre
`implicitfast` de mjlab. Le plan `rustling-dreaming-piglet.md` détaille trois
essais (A1 `implicitfast`, A2 bissection par groupe, A3 pas plus fin), aucun ne
demande de recompiler.

C'est lié : si l'objectif est de réduire l'écart d'actionnement, l'armature et
l'intégrateur sont dans la même boucle que le PD. Ne pas changer les deux à la
fois — deux changements de plant simultanés rendent tout résultat
inattribuable.

## Garde-fous

- **Un entraînement tourne** (`scripts/tools/chain_probes.sh`, sondes P3/P4 puis
  run longue). GPU à ~9.4/11.3 Go. Ne rien lancer sur GPU, ne rien tuer.
- Activer la réplique Elmo veut dire **réentraîner**, pas juste échanger une
  config : la dynamique d'actionnement change, donc la politique aussi.
- Léo pousse sur `leonassim/*`. `origin` de `mc-rtc-superbuild` est le fork de
  Bastien — **ne pas y pousser**.
