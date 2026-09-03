#!/usr/bin/env bash
# T2 : payer la demande de couple avant ecretage. UNE deviation depuis la 6/6.
#
# Depart 2026-09-01_17-45-07 model_4500 -- verifie sur un plateau avant reprise
# (C4) : fell_down 0.0000, sole_height_p90 0.0329->0.0368 et mean_reward
# 78->84 monotones sur 4200-4800, aucun transitoire.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
set -a; source <(grep -E "^RHPS1_" .rhps1_ablation); set +a
export RHPS1_ABLATION="${RHPS1_ABLATION}+demand"
# Forme LINEAIRE bornee, pas le carre. A l'entrainement l'action est
# echantillonnee et la demande n'a pas le meme ordre de grandeur qu'en
# deterministe : fraction au-dessus 0.48 contre 0.022, ratio max 109 contre
# 12.5. Le carre y vaut 14 400 sur un seul joint-pas, donc il paie surtout le
# bruit d'exploration -- dont le deploiement n'a rien, l'ONNX est deterministe.
export RHPS1_DEMAND_POWER=${RHPS1_DEMAND_POWER:-1.0}
export RHPS1_DEMAND_CAP=${RHPS1_DEMAND_CAP:-4.0}
# -0.003 et non -0.04 : la valeur brute passe de 1.1 (carre, deterministe) a
# ~15 (lineaire, entrainement, ~30 articulations). Verifie sur les premieres
# iterations via Episode_Reward/torque_demand, cible -0.9.
export RHPS1_W_DEMAND=${RHPS1_W_DEMAND:--0.003}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
mkdir -p logs/probes
# video_interval est en pas de politique, pas en iterations : 24 pas par
# iteration a 4096 envs, donc 6000 = un clip toutes les 250 iterations.
# 600 pas a 5 ms = 3 s.
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval 6000 --video-length 600 \
  --agent.resume True \
  --agent.load-run 2026-09-01_17-45-07 --agent.load-checkpoint model_4500.pt \
  --agent.max-iterations 2500
# 2500 et non 400 : la run 6/6 a mis 1250 iterations a remonter de son propre
# creux de reprise (-1.14 a l'iteration 3150, -15.8 au creux, +79.9 a 4400).
# Juger une reprise avant ce delai ne mesure pas le changement, ca mesure la
# reprise -- erreur commise ici meme a 13 iterations.
