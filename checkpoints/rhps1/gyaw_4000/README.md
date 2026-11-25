gyaw 4000 (run 2026-11-25_16-17-30): CURRENT BEST FLAT WALKER (2026-11-25), robot-test candidate pending Leo's decision.
Lineage: fw 3000 (forward, scratch) -> gf (actor-only + critic warm-up) -> gh (lr 1e-4) -> gi/gi2 (lr 3e-4) -> gs (swing height)
-> gk (gi2 3000 + swing knee) -> gyaw (gk 3500 + OSEQ_YAW_SS). Recipe gyaw.env, policy.onnx exported under gyaw.env.
mc_mujoco (tardis, flat, slot 22, 50 Hz), no fall on any axis:
  fwd +0.135 m/s, back -0.092, left +0.067, right -0.077, yaw+ +0.24 rad/s, yaw- -0.26
  roll p95 1.1-1.6 deg, steps 0.41-0.46 s, single support 77-95 %, lift 1.6-2.0 cm, knee +6..+9, stop net slip 2.0-4.1 cm
mjlab oseq_eval: 0/8 falls on all 8 commands (fwd 1.04, back 1.04, left 0.74, right 0.93, yaw 0.79/0.75).
Gaps vs slot 18: steps 0.44 s vs 1.03 s, lift 2 cm vs 5.6-6.4 cm, swing knee +7 vs +22.
