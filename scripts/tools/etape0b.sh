#!/usr/bin/env bash
# Etape 0b : config policy 0 + historique 5 + instrumentation + mirror loss +
# paires QP + biais capteurs, SANS la randomisation large.
#
# Test controle d'une hypothese : sous +/-12% de masse, CoM deplace et friction
# variable, apprendre a marcher depuis zero consomme la marge de couple. La
# premiere run tenait 65-70% de saturation sur les jambes, et une politique sans
# reserve echange la translation contre la cadence -- d'ou le martelement a
# 0.13 s. `wide` est le seul element retire.
#
# `rand` reste : c'est lui qui porte les observations biaisees, l'encodeur a
# 0.005 et la vitesse articulaire derivee de ce meme encodeur.
#
# AUCUN curriculum. Il vient apres, une etape a la fois.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+instr+rand+mirror+prox"
export RHPS1_ENC_NOISE=0.005
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/etape0b.train.log 2>&1
