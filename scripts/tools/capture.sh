#!/usr/bin/env bash
# Placement du pied sur le point de capture, en reprise de la run depuis zero.
#
# Reprise AVEC changement de config : sur les cinq reprises de la campagne, les
# trois qui changeaient la config ont recupere (suivi 2.87, 3.02, 3.08) et les
# deux a config identique se sont figees debout sur les talons (1.26, 0.53).
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+capture"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_CAPTURE=-2.0
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 4000 \
  --agent.resume True \
  --agent.load-run 2026-08-29_12-05-02 \
  --agent.load-checkpoint model_10350.pt \
  >> logs/probes/capture.train.log 2>&1
