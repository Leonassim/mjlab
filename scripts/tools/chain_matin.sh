#!/usr/bin/env bash
# Apres la phase 3 : livrables du matin, sans temps mort GPU.
#
#   1. export ONNX du dernier checkpoint de la phase 3
#   2. balayage d'acceptation a 1024 environnements sur ce checkpoint
#   3. balayage sur Base0 model_2100, la reference saine, pour comparer
#   4. rollout d'etalonnage des metriques sur la policy 0 elle-meme
#
# Le 4 est une demande explicite de Leo restee en attente toute la nuit faute
# de memoire GPU : Base0 occupait 10.4 Go sur 11.3. Il passe en dernier parce
# que c'est le seul qui n'a pas besoin d'etre frais pour etre utile.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
P3RUN=2026-08-28_03-09-45
BASE0=2026-08-27_20-27-05
PID=${1:?pid de la phase 3 attendu}
A3="p0+hist5+mirror+masscom+prox+instr+swt+fclr"
A0="p0+hist5+mirror+masscom+prox+instr"

log() { echo "[matin $(date +%H:%M:%S)] $*"; }

for i in $(seq 1 240); do
  [ -f "logs/rsl_rl/rhps1_velocity/$P3RUN/model_3900.pt" ] && break
  kill -0 "$PID" 2>/dev/null || { log "phase 3 terminee ou morte"; break; }
  sleep 30
done
CKPT=$(ls -1t logs/rsl_rl/rhps1_velocity/$P3RUN/model_*.pt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { log "ABANDON : aucun checkpoint"; exit 1; }
CKPT=$(basename "$CKPT")
log "checkpoint retenu $CKPT"

kill "$PID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -9 "$PID" 2>/dev/null
sleep 20
log "GPU libere"

RHPS1_ABLATION="$A3" RHPS1_SWT_TARGET=0.05 \
  .venv/bin/python scripts/tools/export_onnx.py "$P3RUN" "$CKPT" \
  >> logs/probes/export_p3.log 2>&1 && log "ONNX phase 3 exporte" || log "export ECHOUE"

log "balayage phase 3"
RHPS1_ABLATION="$A3" RHPS1_SWT_TARGET=0.05 \
  .venv/bin/python scripts/tools/sweep_eval.py "$P3RUN" "$CKPT" --envs 1024 \
  >> logs/probes/sweep_p3.log 2>&1 && log "balayage phase 3 fait" || log "balayage phase 3 ECHOUE"

log "balayage Base0 (reference)"
RHPS1_ABLATION="$A0" \
  .venv/bin/python scripts/tools/sweep_eval.py "$BASE0" model_2100.pt --envs 1024 \
  >> logs/probes/sweep_base0.log 2>&1 && log "balayage Base0 fait" || log "balayage Base0 ECHOUE"

log "etalonnage policy 0"
RHPS1_ABLATION="p0+instr" \
  .venv/bin/python scripts/tools/sweep_eval.py 2026-07-10_20-59-17 model_9900.pt --envs 1024 \
  >> logs/probes/sweep_policy0.log 2>&1 && log "etalonnage policy 0 fait" || log "etalonnage policy 0 ECHOUE"

log "termine"
