#!/usr/bin/env bash
# Chaine de nuit : Base0 -> export -> phase 1. Le GPU ne s'arrete jamais entre
# deux phases.
#
# Le PID de Base0 est passe en argument et NON cherche par pgrep -f : ce script
# contient lui-meme la ligne de commande d'entrainement, donc un pgrep -f la
# trouverait et le script s'attendrait lui-meme. Ce piege a deja coute 36 min
# de GPU inactif.
set -u
R=/home/lmoussafir/mjlab-rhps1
cd "$R" || exit 1
BASE0=2026-08-27_20-27-05
PID=${1:?pid de Base0 attendu}

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

# 1. attendre le checkpoint 2100, le point de comparaison exact de la reference
for i in $(seq 1 120); do
  [ -f "logs/rsl_rl/rhps1_velocity/$BASE0/model_2100.pt" ] && break
  sleep 20
done
if [ ! -f "logs/rsl_rl/rhps1_velocity/$BASE0/model_2100.pt" ]; then
  log "ABANDON : model_2100.pt jamais apparu, Base0 laisse en vie"; exit 1
fi
log "model_2100.pt present"

# 2. couper Base0 -- transition de phase prevue au plan, porte 533 franchie
kill "$PID" 2>/dev/null
for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -9 "$PID" 2>/dev/null
sleep 20
log "Base0 arrete"

# 3. exporter l'ONNX de reference Base0 (meme ablation que la run, sinon la
#    config d'environnement ne correspond pas au reseau)
RHPS1_ABLATION="p0+hist5+mirror+masscom+prox+instr" \
  .venv/bin/python scripts/tools/export_onnx.py "$BASE0" model_2100.pt \
  >> logs/probes/export_base0.log 2>&1 \
  && log "ONNX Base0 exporte" || log "export ONNX ECHOUE (non bloquant)"

# 4. enchainer la phase 1
log "lancement phase 1"
bash "$R/scripts/tools/phase1.sh"
log "phase 1 terminee"
