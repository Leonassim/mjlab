#!/usr/bin/env bash
# Controle : la config policy 0 SEULE, sur le code d'aujourd'hui.
#
# 0d (p0 + masse +/-3% + link_com +/-1 cm) diverge fortement de la reproduction
# de la policy 0 a iteration egale -- chutes 0.357 contre 0.000, ratio de couple
# 0.659 contre 0.347, air time 0.427 contre 0.154. La randomisation masse/CoM
# est le seul ecart de configuration, MAIS la reproduction tournait sur l'arbre
# de juillet avec son venv epingle. Cette run separe les deux causes : meme
# code, meme venv, aucune randomisation ajoutee.
#
# Pas d'instr : les six metriques utiles a la comparaison (fell_down,
# error_vel_xy, torque_limit_ratio_mean, landing_vel_mean, air_time_mean,
# track_linear_velocity) sont deja dans repro.log sans elle, donc l'ajouter
# serait un ecart de plus pour rien.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0"
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 12000 \
  >> logs/probes/ctrl_p0.train.log 2>&1
