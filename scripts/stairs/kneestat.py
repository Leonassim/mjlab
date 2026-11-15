"""kneestat.py: loaded-knee flexion, step length, falls per level (deterministic); was ftrace: per-0.1 s trace of pinned stair envs; prints the last 3 s of each fall + one survivor.
old: stairs_bench.py <run> <ckpt> [label]: deterministic stairs bench, one line per height.

Envs are pinned: each stair env gets a fixed level (step height) and a fixed mode
(climb / descend), several per cell; flat envs check the copy. 12 s per episode,
no resets (a terminated env counts as a fall and stops counting).

  BENCH_LEVELS=1,3,5,7,9 BENCH_PER=4 uv run python stairs_bench.py <run> <ckpt>
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
# Overrides for running chains: KEY=VAL lines in bench_override next to this script.
_ov = Path(__file__).with_name("bench_override")
if _ov.exists() and not os.environ.get("BENCH_NO_OVERRIDE"):
  for _l in _ov.read_text().split():
    _k, _v = _l.split("=", 1)
    os.environ[_k] = _v
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ck
LEVELS = [int(x) for x in os.environ.get("BENCH_LEVELS", "0,1,2,4,9").split(",")]
PER = int(os.environ.get("BENCH_PER", "4"))
NFLAT = int(os.environ.get("BENCH_FLAT", "16"))
T = float(os.environ.get("BENCH_T", "16"))
dev = os.environ.get("BENCH_DEV", "cpu")
torch.manual_seed(int(os.environ.get("BENCH_SEED", "0")))

cells = [(lv, climb) for lv in LEVELS for climb in (True, False)]
NST = len(cells) * PER
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = NST + NFLAT


def _fall_all(env):
  # Real falls on every env: the copy's tracking terminations fire on reference drift in long episodes.
  robot = env.scene["robot"]
  ground = env.scene["stairs_ground"].data.hit_pos_w[:, 2, 2]
  low = robot.data.root_link_pos_w[:, 2] - ground < 0.5
  tilt = robot.data.projected_gravity_b[:, 2] > -0.55
  return low | tilt | (env.scene["stairs_body"].data.found.sum(-1) > 0)


cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
cfg.terminations["stairs_fall"].func = _fall_all
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=float(os.environ["CLIP_ACTIONS"]) if os.environ.get("CLIP_ACTIONS") else None)
u = env.unwrapped
st = u._stairs
terrain = u.scene.terrain

# Pin types/levels: the first NST envs on the 27 cm column, the rest on the flat column.
kinds = cfg.commands["motion"].stairs_kinds
col_st = kinds.index("up")
col_flat = kinds.index("flat")
lv = torch.tensor([c[0] for c in cells for _ in range(PER)] + [0] * NFLAT, device=u.device)
ty = torch.tensor([col_st] * NST + [col_flat] * NFLAT, device=u.device)
terrain.terrain_levels[:] = lv
terrain.terrain_types[:] = ty
terrain.env_origins[:] = terrain.terrain_origins[lv, ty]
st.kind = torch.where(ty == col_st, 1, 0)
st.is_stairs = st.kind > 0
st.tread = torch.where(st.is_stairs, torch.tensor(cfg.commands["motion"].stairs_treads, device=u.device)[ty], 0.0)
st.nsteps = torch.where(st.is_stairs, ST.NSTEPS, 0)
want_climb = torch.tensor([c[1] for c in cells for _ in range(PER)] + [False] * NFLAT, device=u.device)
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True}, strict=True, map_location=dev)
pol = runner.get_inference_policy(device=dev)
u.reset()
# Mode pin: respawn every stair env whose drawn mode differs, until all match (bounded).
cmd = u.command_manager.get_term("motion")
for _ in range(30):
  bad = (st.is_stairs & (st.climb != want_climb)).nonzero().flatten()
  if len(bad) == 0:
    break
  cmd._in_reset = True
  cmd._resample_command(bad)
  cmd._in_reset = False
u.sim.forward()
assert bool((st.climb[st.is_stairs] == want_climb[st.is_stairs]).all()), "mode pin failed"
obs = env.get_observations()




import math
steps = int(T / u.step_dt)
robot = u.scene["robot"]
tm = u.termination_manager
fids = st.foot_ids
N = u.num_envs
kn = [robot.joint_names.index(n) for n in ("L_KNEE_P", "R_KNEE_P")]
alive = torch.ones(N, dtype=torch.bool, device=u.device)
kl = [[] for _ in range(N)]
was = torch.zeros(N, 2, dtype=torch.bool, device=u.device)
last = torch.full((N, 2, 2), float("nan"), device=u.device)
steplen = [[] for _ in range(N)]
for i in range(steps):
  with torch.no_grad():
    obs, rew, dones, extras = env.step(pol(obs))
  loaded = st.load > 0.1 * ST.BW
  q = robot.data.joint_pos[:, kn]
  fp = robot.data.body_link_pos_w[:, fids, :2]
  td = loaded & ~was
  for e, f in td.nonzero().tolist():
    if alive[e]:
      if not math.isnan(float(last[e, f, 0])):
        steplen[e].append(float(((fp[e, f] - last[e, f]) * st.face[e]).sum()))
      last[e, f] = fp[e, f]
  was = loaded
  m = loaded & alive[:, None]
  for e, f in m.nonzero().tolist():
    kl[e].append(math.degrees(float(q[e, f])))
  alive &= ~tm.terminated
import numpy as np
for lvl in sorted(set(int(x) for x in lv.tolist())):
  for cl in (True, False):
    es = [e for e in range(NST) if int(lv[e]) == lvl and bool(want_climb[e]) == cl]
    if not es: continue
    k = np.array([x for e in es for x in kl[e]]); sl = np.array([x for e in es for x in steplen[e] if x > 0.02])
    fell = sum(1 for e in es if not bool(alive[e]))
    print(f"{label} lvl {lvl} {'monte' if cl else 'desc '}: chutes {fell}/{len(es)} | genou en appui p50 {np.percentile(k,50):.0f} p90 {np.percentile(k,90):.0f} p99 {np.percentile(k,99):.0f} deg, >45 deg {100*(k>45).mean():.0f}% | pas (avance du pied) med {100*np.median(sl):.0f} p90 {100*np.percentile(sl,90):.0f} cm n {len(sl)}")
