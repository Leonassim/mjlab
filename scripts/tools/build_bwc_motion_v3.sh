#!/usr/bin/env bash
# docs/bwc_motion_v3.npz from Leo's 2026-10-09 BWC session in mc_mujoco (all six
# directions). Long standing stretches are cut: each clip keeps 10 s of standing
# before and after one walk. The two diagonal forward walks (828 s, 887 s) are
# left out. Windows in seconds from the log start.
#   bash scripts/tools/build_bwc_motion_v3.sh <log.bin> <workdir>
set -eu
LOG=$1; W=$2; mkdir -p "$W"
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES= uv run python -c "
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
c = load_env_cfg('Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start', play=True); c.scene.num_envs = 1
c.commands['motion'].motion_file = 'docs/bwc_motion_v2.npz'
e = ManagerBasedRlEnv(c, device='cpu')
open('$W/model_order.txt', 'w').write(','.join(e.scene['robot'].joint_names))"
python3 scripts/tools/bwc_log_to_csv.py "$LOG" "$W/full.csv" --model-order "$W/model_order.txt" --keep-still --min-speed 0
CLIPS="A_forward:224:263 B_left:412:487 C_backward:516:555 D_right:609:684 E_yaw_left:710:754 F_yaw_right:764:807"
for c in $CLIPS; do
  IFS=: read -r name a b <<<"$c"
  sed -n "$((a * 200 + 1)),$((b * 200))p" "$W/full.csv" > "$W/$name.csv"
  uv run python src/mjlab/scripts/csv_to_npz.py --input-file "$W/$name.csv" --output-name "$W/$name.npz" \
    --robot rhps1 --input-fps 200 --output-fps 200 --device cuda:0
done
python3 - "$W" $CLIPS <<'PY'
import sys, numpy as np
w = sys.argv[1]; names = [c.split(":")[0] for c in sys.argv[2:]]
cl = [dict(np.load(f"{w}/{n}.npz")) for n in names]
out = {k: (np.concatenate([c[k] for c in cl]) if cl[0][k].ndim > 1 else cl[0][k]) for k in cl[0]}
out["clip_starts"] = np.cumsum([0] + [len(c["joint_pos"]) for c in cl[:-1]])
np.savez_compressed("docs/bwc_motion_v3.npz", **out)
print("docs/bwc_motion_v3.npz", len(out["joint_pos"]), "frames, clips at", out["clip_starts"])
PY
