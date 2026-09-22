#!/usr/bin/env bash
# Banc de deploiement autonome sous mc_mujoco, sans interface.
#   mcbench.sh <index> <vx> <vy> <wz> [secondes]
#
# mc_rtc journalise a 200 Hz SANS plafond : un run laisse libre a ecrit 29 Go en
# 50 min le 2026-09-15. mc_mujoco est donc lance dans son PROPRE GROUPE de
# processus (setsid) et c'est le GROUPE ENTIER qui est tue -- un `timeout` simple
# ne tuait que l'enveloppe et laissait mc_mujoco orphelin.
set -u
# Never touch an mc_mujoco someone else is running.
pgrep -x mc_mujoco >/dev/null && { echo "mc_mujoco deja en cours : banc saute"; exit 3; }
IDX=$1; VX=$2; VY=$3; WZ=$4; SECS=${5:-15}
SCR=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad
OVL=$SCR/bench_overlay.yaml
cat > "$OVL" <<YAML
NewRLQPController:
  default_policy_index: $IDX
YAML
# Le banc passe par l'environnement : une section de recouvrement ne fusionne pas
# avec la configuration propre du controleur (2026-09-15).
export RLQP_BENCH_POLICY_INDEX=$IDX
export RLQP_BENCH_AUTO_ARM=3.0
export RLQP_BENCH_VELOCITY="$VX $VY $WZ"
export MC_RTC_CONTROLLER_CONFIG="${OVL}${MC_RTC_CONTROLLER_CONFIG:+:${MC_RTC_CONTROLLER_CONFIG}}"

touch "$SCR/mcbench.start"
setsid /home/lmoussafir/install/bin/mc_mujoco --sync --without-visualization --without-mc-rtc-gui \
  -f /home/lmoussafir/install/etc/mc_rtc_superbuild_mujoco.yaml > "$SCR/mcbench.log" 2>&1 &
PGID=$!
# Le demarrage (maillages, init mc_rtc) prend ~128 s : attendre que mc_rtc cree
# son journal plutot que de deviner un delai. Le fichier apparait a l'init du
# controleur, donc sa presence signale que la simulation tourne vraiment.
for _ in $(seq 1 60); do
  sleep 5
  find /home/lmoussafir/logs/mc_rtc -name "mc-control-NewRLQPController-2*.bin" -newer "$SCR/mcbench.start" | grep -q . && break
  pgrep -x mc_mujoco >/dev/null || { echo "mc_mujoco est mort avant l init"; break; }
done
find /home/lmoussafir/logs/mc_rtc -name "mc-control-NewRLQPController-2*.bin" -newer "$SCR/mcbench.start" | grep -q . \
  && echo "controleur initialise, $SECS s de mesure" \
  || echo "ATTENTION : aucun journal apres 5 min"
sleep "$SECS"
kill -TERM -$PGID 2>/dev/null; sleep 3
kill -KILL -$PGID 2>/dev/null; sleep 1
# Rien ne doit survivre : le point qui a coute 29 Go.
REST=$(pgrep -x mc_mujoco | tr '\n' ' ')
[ -n "$REST" ] && echo "ATTENTION mc_mujoco encore vivant (pid $REST) : pas tue, peut appartenir a Leo"
echo "--- armement ---"; grep -c "BANC : armement" "$SCR/mcbench.log" 2>/dev/null
echo "--- log produit ---"; ls -la /home/lmoussafir/logs/mc_rtc/mc-control-NewRLQPController-*.bin 2>/dev/null | tail -1
