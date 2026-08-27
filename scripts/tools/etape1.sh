#!/usr/bin/env bash
# Etape 1 : la configuration qui a tourne sur le robot, plus la masse a +/-3%.
#
# La reference n'est PAS `p0` brut ni la reproduction du worktree de juillet --
# dont l'ONNX n'existe nulle part dans l'historique git de rl_controller. C'est
# rhps1_velocity_lift_it2100.onnx, commit 0d15f9f du 2026-08-21, index 1, issu
# de la run 2026-08-21_11-47-31 avec l'ablation p0+rand+lift.
#
# Ce que `lift` change et pourquoi p0 brut se comporte mal sans lui :
#   min_foot_height   min_height 0.08 -> 0.02. A 0.08 le terme coute 0.063 d'un
#                     maximum de 0.069, soit 92% : une taxe plate sur le fait
#                     d'etre en l'air, donc sur l'air time voulu.
#   foot_swing_height target 0.15 -> 0.03. A 0.15 l'erreur relative au carre
#                     sature a 1.0, le terme est une constante.
#   air_time          threshold_max epingle a 0.25 au lieu du curriculum qui le
#                     promene 0.1 -> 0.3 -> 0.5.
#
# Seul ecart avec la run deployee : link_inertia +/-5% -> +/-3%, la demande de
# Leo. instr est garde (poids 1e-9, mesure seulement).
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+rand+lift+masscom+instr"
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/base.train.log 2>&1
