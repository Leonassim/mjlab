#!/usr/bin/env bash
# Controle horaire COMPLET : courbes + banc de deploiement.
# Le banc est le critere principal -- les metriques de tracking ont laisse passer
# une politique incapable d'obeir a une commande (2026-09-15).
set -u
cd /home/lmoussafir/mjlab-rhps1
R=$(ls -t logs/rsl_rl/rhps1_tracking | head -1)
C=$(ls logs/rsl_rl/rhps1_tracking/$R/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
[ -z "$C" ] && { echo "aucun checkpoint"; exit 0; }
RHPS1_POSTURE_K=1600 RHPS1_POSTURE_K_HI=0 CUDA_VISIBLE_DEVICES="" timeout 2400 uv run python scripts/tools/deploy_bench.py "$R" "model_$C.pt" \
  --secs 12 --envs 16 > /tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/deploy_out.txt 2>&1
grep -aE "mode|immobile|avant|gauche|droite|rotation|DEBOUT" /tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/deploy_out.txt | tail -7
echo "  (reference : avant 0.17 m/s soit ~2.0 m en 12 s ; policy 0 sous mc_mujoco 0.109 m/s)"
RHPS1_POSTURE_K=1600 RHPS1_POSTURE_K_HI=0 CUDA_VISIBLE_DEVICES="" timeout 1500 uv run python scripts/tools/rsi_check.py "$R" "model_$C.pt" 2>&1 | tail -3
# mc_mujoco: export latest checkpoint into slot 2, 20 s forward.
O=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/hourly_$C.onnx
CUDA_VISIBLE_DEVICES="" MJLAB_TASK=Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start MJLAB_LOG_ROOT=logs/rsl_rl/rhps1_tracking \
  MJLAB_MOTION_FILE=docs/bwc_motion_v2.npz timeout 900 uv run python scripts/tools/export_onnx.py "$R" "model_$C.pt" "$O" > /dev/null 2>&1 \
  && ! pgrep -x mc_mujoco > /dev/null \
  && uv run python scripts/tools/check_slot_scales.py "$O" 4 2>&1 | tail -1 | tee /dev/stderr | grep -q "== ONNX" \
  && cp /home/lmoussafir/install/share/NewRLQPController/policy/rhps1_bwc_x10.onnx /tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/slot4_backup.onnx \
  && cp "$O" /home/lmoussafir/install/share/NewRLQPController/policy/rhps1_bwc_x10.onnx \
  && { timeout 600 /tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/mcbench.sh 4 1.0 0 0 20 > /dev/null 2>&1; true; } \
  && python3 scripts/tools/mc_log_summary.py --export=/tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/mc_gait.npz 2>&1 | tail -1 \
  && uv run python scripts/tools/mc_gait.py /tmp/claude-1002/-home-lmoussafir-mc-rtc-superbuild/6f3e2030-bd2b-4492-8c6b-6737c1f6af5e/scratchpad/mc_gait.npz 2>&1 | tail -1
RHPS1_POSTURE_K=1600 RHPS1_POSTURE_K_HI=0 CUDA_VISIBLE_DEVICES="" timeout 1500 uv run python scripts/tools/fall_probe.py "$R" "model_$C.pt" noise_settle 2>&1 | grep -a "debout" &
RHPS1_POSTURE_K=1600 RHPS1_POSTURE_K_HI=0 CUDA_VISIBLE_DEVICES="" timeout 1500 uv run python scripts/tools/fall_probe.py "$R" "model_$C.pt" settle 2>&1 | grep -a "20s debout\|pieds"
wait
