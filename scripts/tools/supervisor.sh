#!/usr/bin/env bash
# Superviseur : le GPU ne reste jamais libre.
#
# Attend la fin de l'entrainement en cours, exporte, balaie, puis relance une
# continuation depuis le dernier checkpoint PERIODIQUE (jamais le save de fin :
# ph4 et ph5 en sont reparties et n'ont jamais recupere au-dessus de 1.10
# contre 2.94 pour une reprise depuis un periodique).
#
# Le PID est passe en argument, jamais cherche par pgrep -f : ce script contient
# la ligne de commande d'entrainement et se trouverait lui-meme.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
PID=${1:?pid de l entrainement attendu}
ABL=${2:?ablation attendue}
ROUNDS=${3:-3}
log(){ echo "[sup $(date +%m-%d_%H:%M:%S)] $*"; }

for round in $(seq 1 "$ROUNDS"); do
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
  sleep 20
  RUN=$(ls -1dt logs/rsl_rl/rhps1_velocity/*/ | head -1); RUN=$(basename "$RUN")
  # dernier checkpoint PERIODIQUE : multiple de 150
  CKPT=$(ls -1 logs/rsl_rl/rhps1_velocity/$RUN/model_*.pt 2>/dev/null \
         | sed 's|.*/model_||; s|\.pt||' | sort -n \
         | awk '$1 % 150 == 0' | tail -1)
  if [ -z "$CKPT" ]; then log "ABANDON : aucun checkpoint periodique dans $RUN"; exit 1; fi
  log "tour $round : $RUN model_$CKPT.pt"

  RHPS1_ABLATION="$ABL" RHPS1_SWT_TARGET=0.05 \
    .venv/bin/python scripts/tools/export_onnx.py "$RUN" "model_$CKPT.pt" \
    >> logs/probes/sup_export.log 2>&1 && log "onnx exporte" || log "export echoue"

  RHPS1_ABLATION="$ABL" RHPS1_SWT_TARGET=0.05 \
    .venv/bin/python scripts/tools/sweep_eval.py "$RUN" "model_$CKPT.pt" --envs 1024 \
    >> "logs/probes/sup_sweep_${RUN}_${CKPT}.log" 2>&1 && log "balayage fait" || log "balayage echoue"

  log "relance depuis $RUN model_$CKPT.pt"
  RHPS1_ABLATION="$ABL" RHPS1_SWT_TARGET=0.05 \
  WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300 \
  .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
    --env.scene.num-envs 4096 --video True \
    --video-interval 12000 --video-length 600 \
    --agent.max-iterations 3000 \
    --agent.resume True --agent.load-run "$RUN" \
    --agent.load-checkpoint "model_$CKPT.pt" \
    >> logs/probes/supervisor.train.log 2>&1 &
  PID=$!
  log "relance pid $PID"
  sleep 120
done
log "termine"
