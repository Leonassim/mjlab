# vis3c model_32500: stairs vision student (PPO), not a BWC copy

- Run `2026-11-19_11-31-06`, task `Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start`, motion `docs/bwc_motion_v3.npz`, 3584 envs, 50 Hz (decimation 8).
- Code: commit `cad06081c` (vision_map GATE_SKIP / CLIP).
- Recipe: `scripts/stairs/s50e.env` plus `vis3c_extra.env` (RHPS1_VIS=1, real Azure pose, map failures: outages, persons, ray drop, phantoms, gate skip, clip 0.6 m). Level file `vis3c_lvl.pt`.
- Launch: `set -a; source vis3c_extra.env; set +a; ITERS=15000 bash scripts/stairs/train_s50.sh vis3c 2026-11-18_22-48-10 model_24000.pt` (entropy 0, ramps frozen).
- Lineage: s50e 15000 -> vis1 -> vis2 22500/24000 -> vis3c (24000 -> 32719, stopped 2026-11-20 20:29). Target 39000.
- Actor reads the onboard height map (vision_map.py); deploys as rl_controller slot 23 (obs_format 12).

Gates (scripts/stairs/vis_gate_loop.sh, 32 envs x 24 s, falls / arrivals):

| it | near | far | rob (map failures) |
|---|---|---|---|
| 25000 | 1 / 27 | 2 / 24 | 14 / 15 |
| 28500 | 0 / 27 | 1 / 26 | 4 / 18 |
| 30000 | 0 / 27 | 1 / 25 | 6 / 19 |
| 31000 | - | - | 4 / 21 |
| 31500 | - | - | 3 / 23 |
| 32000 | - | - | 5 / 21 |
| 32500 | 0 / 27 | 0 / 26 | 8 / 20 |
