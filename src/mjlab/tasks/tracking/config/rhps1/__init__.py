from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import rhps1_flat_tracking_env_cfg
from .rl_cfg import rhps1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1",
  env_cfg=rhps1_flat_tracking_env_cfg(),
  play_env_cfg=rhps1_flat_tracking_env_cfg(play=True),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Le robot ne publie pas d'estimation de base fiable en deploiement : mc_rtc la
# reconstruit par MCWaiko, et la mesure du 2026-09-10 donne 0.43 m/s d'erreur RMS
# sur vx pendant une chute. Cette variante retire motion_anchor_pos_b et
# base_lin_vel de l'observation de l'acteur, comme celle du G1.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-No-State-Estimation",
  env_cfg=rhps1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=rhps1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 1 : copier la trajectoire, sans randomisation. Voir le docstring de
# rhps1_flat_tracking_env_cfg.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Copy",
  env_cfg=rhps1_flat_tracking_env_cfg(randomize=False),
  play_env_cfg=rhps1_flat_tracking_env_cfg(randomize=False, play=True),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 1b : l'eleve reprend la main. La distillation pure plafonne a MSE 0.88 et
# 7/16 chutes sur 20 s -- il lui manque motion_anchor_pos_b, l'ecart accumule a la
# reference, qui est la boucle de rattrapage de l'experte. Plutot que d'inventer
# une observation de substitution, on le laisse apprendre son propre rattrapage :
# meme plant, memes recompenses, acteur a 99 dims, critique privilegiee.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Copy-Student",
  env_cfg=rhps1_flat_tracking_env_cfg(randomize=False, student_actor=True),
  play_env_cfg=rhps1_flat_tracking_env_cfg(
    randomize=False, student_actor=True, play=True
  ),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 2 : le meme eleve, randomisation reintroduite. Les plages de policy 0 ne
# sont pas une marge de robustesse, elles portent le transfert (memoire
# rhps1-train-play-gap-resolved) -- donc on les remet avant tout deploiement.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Student",
  env_cfg=rhps1_flat_tracking_env_cfg(student_actor=True),
  play_env_cfg=rhps1_flat_tracking_env_cfg(student_actor=True, play=True),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 3 : format V3 a 126 dims (case 0 du controleur) et noyau des pieds
# resserre a 0.015. Voir les commentaires de deploy_v3 dans env_cfgs.py.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Student-V3",
  env_cfg=rhps1_flat_tracking_env_cfg(student_actor=True, deploy_v3=True),
  play_env_cfg=rhps1_flat_tracking_env_cfg(
    student_actor=True, deploy_v3=True, play=True
  ),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 4 : V3 126 dims + terme de cadence. Le deficit qui reste apres la copie
# de trajectoire est le NOMBRE de pas (114 cycles contre 160), pas leur hauteur.
# Voir _contact_schedule_match dans env_cfgs.py.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Student-V3-Cadence",
  env_cfg=rhps1_flat_tracking_env_cfg(
    student_actor=True, deploy_v3=True, cadence_term=1.0
  ),
  play_env_cfg=rhps1_flat_tracking_env_cfg(
    student_actor=True, deploy_v3=True, cadence_term=1.0, play=True
  ),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 4 : copier la marche AVANT de randomiser. La phase 2 a montre que la
# randomisation fait tomber le pic par cycle de 7.96 a 2.28 cm -- un pied haut
# n'absorbe pas une poussee. On fixe donc d'abord la cadence sans randomisation,
# puis on randomise avec le terme deja en place.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Copy-V3-Cadence",
  env_cfg=rhps1_flat_tracking_env_cfg(
    randomize=False, student_actor=True, deploy_v3=True, cadence_term=1.0
  ),
  play_env_cfg=rhps1_flat_tracking_env_cfg(
    randomize=False, student_actor=True, deploy_v3=True, cadence_term=1.0, play=True
  ),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# PHASE 5 : commande directionnelle + departs ARRETES en posture de deploiement.
# Sans ces departs, la politique n'apprend qu'a continuer un mouvement : mesure
# du 2026-09-15, la distance parcourue sur commande tombe de 0.49 a 0.04 m sur
# trois lectures pendant que recompense et survie montent.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start",
  env_cfg=rhps1_flat_tracking_env_cfg(
    randomize=False, student_actor=True, deploy_v3=True, cadence_term=1.0,
    standing_start=0.3,
  ),
  play_env_cfg=rhps1_flat_tracking_env_cfg(
    randomize=False, student_actor=True, deploy_v3=True, cadence_term=1.0,
    standing_start=0.3, play=True,
  ),
  rl_cfg=rhps1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
