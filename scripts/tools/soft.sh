#!/usr/bin/env bash
# Impact : ramener le plafond de impact_vel de 0.45 a 0.20, sans toucher au
# poids. Vise O2.
#
# Leo mesure 0.2 m/s en z et 0.35 en absolu dans mc_mujoco ; le balayage
# deterministe dit 0.179 et l'entrainement 0.088. Le plafond a 0.45 est 2.5 a
# 3.7 fois au-dessus de tout ca, donc le terme paie une quasi-constante et
# descendre ne rapporte rien -- c'est la contrainte C6 sur les plafonds non
# clampes, qui doivent se poser JUSTE au-dessus de la mesure.
#
# Le poids reste a -0.5, deliberement. C7 dit qu'une penalite d'atterrissage
# amplifiee se paie en n'atterrissant plus, verifie cinq fois dont une malgre
# l'horloge. Baisser le plafond fait mordre le cout existant sans l'augmenter.
#
# 0.20 est aussi juste au-dessus des 0.158 de la policy 0, la demarche qui
# atterrissait assez doucement sur le materiel.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
export RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+swt+fclr+comshift+clock+steplen+swingbonus+softland"
export RHPS1_SWT_TARGET=0.05
export RHPS1_W_SWING=0.0
export RHPS1_W_SWINGBONUS=6.0
export RHPS1_SWINGBONUS_H=0.05
export RHPS1_W_CLOCK=2.0
export RHPS1_STEP_TARGET=0.03
export RHPS1_W_STEPLEN=2.0
export RHPS1_IMPACT_LIMIT=0.20
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
.venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 12000 --video-length 600 \
  --agent.max-iterations 8000 \
  --agent.resume True \
  --agent.load-run 2026-08-31_22-19-33 \
  --agent.load-checkpoint model_3300.pt \
  >> logs/probes/soft.train.log 2>&1
