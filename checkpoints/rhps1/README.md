# Checkpoints RHPS1 de référence

Chaque dossier contient le checkpoint rsl-rl (`model_*.pt`), la configuration exacte de son entraînement
(`agent.yaml`, `env.yaml`) et, quand il existe, l'ONNX déployable (`policy.onnx`). Pour le réutiliser avec les
scripts, copier le dossier sous `logs/rsl_rl/rhps1_tracking/<nom du run>/` (nom donné ci-dessous) ; pour un test
mc_mujoco, l'ONNX suffit (voir [`scripts/stairs/README.md`](../../scripts/stairs/README.md)).

Banc mc_mujoco « 12 essais » : marches de 4, 5 et 6 cm × départ de la commande 2, 3, 4 ou 5 s après l'armement,
commande avant jusqu'au bout (monter, traverser le palier, redescendre). « Palier » = bassin au-delà du premier
mètre de l'escalier ; « descente » = arrivé en bas de l'autre côté.

| Dossier | Run d'origine | Fréquence | Ce que c'est | Banc 12 essais mc_mujoco |
|---|---|---|---|---|
| `slot18_flat_walker` | `2026-10-29_06-55-45` model_12499 | 200 Hz | Marche à plat par copie du BWC, référence sur le vrai robot (slot 18). Observations V3 (126). | — (pas d'escaliers) |
| `stairs_init_slot20` | `2026-11-06_stairs_init` model_13498 | 200 Hz | Slot 20 (slot 18 réentraîné sur la plante cheville KI 0,02) avec les 325 entrées du scan ajoutées à poids nuls : point de départ de toute la lignée escaliers 200 Hz. | aveugle, 0 arrivée |
| `slot22_rl48b` | `2026-11-12_21-20-08` model_31500 | 200 Hz | RL escaliers, installé dans le slot 22 de rl_controller ; premier à monter 2 cm puis 5 cm (environ une fois sur deux à l'écran). | palier 2/12, sans chute 1/12, genou ≥ 105 N·m 25 % du temps |
| `rl51_200hz` | `2026-11-13_23-14-18` model_37000 | 200 Hz | + coût de couple par joint et terminaison sur saturation prolongée. | palier 8/12 (4 cm 4/4), sans chute 2/12, genou 18 % |
| `rl54_200hz` | `2026-11-14_05-51-13` model_42500 | 200 Hz | + coût sur le genou trop plié en appui, curriculum plafonné à 6 cm. | palier 5/12, sans chute 4/12, genou 13 % |
| `ap7_200hz_approach` | `2026-11-14_19-41-37` model_47000 | 200 Hz | Tâche d'approche (départ sur le plat à 0,5-1,5 m, ±30°, arrêts sur l'escalier), mjlab fusionné. Trop prudent sans bruit d'exploration. | palier 0/12, sans chute 4/12 |
| `s50_50hz_scratch` | `2026-11-14_22-21-43` model_3000 | 50 Hz | Départ de zéro à 50 Hz (plat RL, envs debout, approche). | palier 0/12, sans chute 7/12 |
| `s50e_50hz` | `2026-11-15_10-23-12` model_3500 | 50 Hz | s50 repris avec entropie 0. **Meilleur résultat à ce jour.** | **palier 7/12, sans chute 10/12, 2 descentes complètes, 5 cm : palier 4/4** ; genou 5 % |

Rollouts déterministes mjlab (32 envs × 24 s, départ proche) : rl54 42750 20 arrivées / 26 chutes / 0,066 m/s ;
s50 3000 29 / 19 / 0,060 ; s50e 3500 30 / 11 / 0,053.

Pas encore testés sur le vrai robot : slot 22 et les suivants. Les politiques 50 Hz demandent un slot
`policy_step_size: 0.02` dans rl_controller.
