#!/usr/bin/env bash
# Phase 1 -> phase 2, sans temps mort GPU.
#
# Coupe la phase 1 a l'iteration 3150 et non a son terme naturel (6300) : la
# reprise ajoute max-iterations au checkpoint charge, donc la phase 1 durerait
# 8 h a elle seule. 1050 iterations suffisent pour juger contact_balance et
# laissent la nuit a deux autres phases.
#
# PID passe en argument, jamais cherche par pgrep -f : ce script contient la
# ligne de commande d'entrainement et se trouverait lui-meme.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
P1RUN=2026-08-28_00-47-58
PID=${1:?pid de la phase 1 attendu}
CKPT=model_3150.pt

log() { echo "[chain2 $(date +%H:%M:%S)] $*"; }

for i in $(seq 1 240); do
  [ -f "logs/rsl_rl/rhps1_velocity/$P1RUN/$CKPT" ] && break
  kill -0 "$PID" 2>/dev/null || { log "phase 1 morte avant $CKPT"; break; }
  sleep 30
done
if [ ! -f "logs/rsl_rl/rhps1_velocity/$P1RUN/$CKPT" ]; then
  log "ABANDON : $CKPT absent"; exit 1
fi
log "$CKPT present"

kill "$PID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -9 "$PID" 2>/dev/null
sleep 20
log "phase 1 arretee"

RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr+cbal" RHPS1_W_CBAL=0.5 \
  .venv/bin/python scripts/tools/export_onnx.py "$P1RUN" "$CKPT" \
  >> logs/probes/export_p1.log 2>&1 \
  && log "ONNX phase 1 exporte" || log "export ECHOUE (non bloquant)"

log "lancement phase 2"
bash "$R/scripts/tools/phase2.sh" "$P1RUN" "$CKPT"
log "phase 2 terminee"
