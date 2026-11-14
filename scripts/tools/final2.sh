#!/usr/bin/env bash
# Poursuite de la run propre, depuis model_3150 -- le checkpoint qui a franchi
# le seuil de lever de pied au balayage deterministe.
#
# Balayage de model_3150, 1024 envs, 11 commandes :
#   lever de pied 0.0334 (seuil 0.030)  OK, premiere fois de la campagne
#   couples jambes 0.0208, haut 0.0000  OK
#   chutes 0.0010, pieds a plat 0.0277  OK
#   impact 0.1793 (seuil 0.160)         ECHEC de 12%, mais MIEUX que la
#                                       policy 0 qui mesure 0.1919
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=0.0
export RHPS1_W_SWINGBONUS=6.0
export RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_15-11-13 \
  --agent.load-checkpoint model_3150.pt \
  >> logs/probes/final2.train.log 2>&1
