"""Diff a controller slot's action_scale against an exported ONNX, by joint name.

  uv run python scripts/tools/check_slot_scales.py <policy.onnx> <slot>
"""
import sys

import onnx
import yaml

md = {p.key: p.value for p in onnx.load(sys.argv[1]).metadata_props}
names = md["joint_names"].split(",")
scale = [float(x) for x in md["action_scale"].split(",")]
y = yaml.safe_load(open("/home/lmoussafir/install/lib/mc_controller/etc/NewRLQPController.yaml"))
slot = y["policies"][int(sys.argv[2])]["action_scale"]
bad = [(n, s, slot.get(n)) for n, s in zip(names, scale) if slot.get(n) is None or abs(s - slot[n]) > 1e-6 * max(1.0, s)]
print("echelles slot %s == ONNX" % sys.argv[2] if not bad else "ECHELLES DIFFERENTES slot %s : %s" % (sys.argv[2], bad))
