#!/usr/bin/env bash
# Lance un entrainement ET arme son veilleur. Une seule commande, parce que les
# separer m'a fait laisser quatre runs sans surveillance le 2026-09-11.
#
#   JALON=3013 scripts/tools/launch_watched.sh <nom> -- <args de train>
#
# Le nom sert au fichier de log. Tout ce qui suit `--` va a `train`.
set -u
NAME=$1; shift
[ "${1:-}" = "--" ] && shift
DIR=${SCRATCH:-/tmp}/rhps1_runs; mkdir -p "$DIR"
LOG="$DIR/$NAME.log"
JALON=${JALON:-3013}

cd /home/lmoussafir/mjlab-rhps1 || exit 1
# Refuser de demarrer si un entrainement tourne deja : il tiendrait le GPU et le
# nouveau prendrait un OOM, pendant que la detection ci-dessous confondrait les
# deux. Arrive le 2026-09-11 : un kill sur une variable vide a laisse l'ancien
# run vivant 2 h de plus et le script a "detecte" celui-la comme trainer.
for p in $(ps -o pid= -e); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *".venv/bin/python"*"bin/train"*)
    echo "REFUS : un entrainement tourne deja (pid $p). L'arreter d'abord." >&2
    exit 1;; esac
done

setsid nohup uv run train "$@" > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null

# Attendre que le VRAI processus existe, pas l'enveloppe uv qui exec puis meurt.
for _ in $(seq 1 90); do
  sleep 10
  grep -q "Traceback" "$LOG" && { echo "ECHEC au demarrage :"; tail -5 "$LOG"; exit 1; }
  for p in $(ps -o pid= -e); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in *".venv/bin/python"*"bin/train"*) TP=$p; break 2;; esac
  done
done
[ -z "${TP:-}" ] && { echo "aucun trainer detecte apres 15 min"; exit 1; }

echo "trainer $TP   log $LOG   jalon $JALON"
echo "$TP" > "$DIR/$NAME.pid"
echo "$LOG" > "$DIR/$NAME.logpath"
