"""ftrace.py: per-0.1 s trace of pinned stair envs; prints the last 3 s of each fall + one survivor.
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



steps = int(T / u.step_dt)
robot = u.scene["robot"]
tm = u.termination_manager
fids = st.foot_ids
N = u.num_envs
span = ST.PLATFORM / 2 + (ST.NSTEPS - 1) * st.tread
kn = [i for i, n in enumerate(robot.joint_names) if "KNEE" in n]
print("knee joints", [robot.joint_names[i] for i in kn], "feet", fids)
alive = torch.ones(N, dtype=torch.bool, device=u.device)
log = {e: [] for e in range(NST)}
every = max(1, int(0.1 / u.step_dt))
for i in range(steps):
  with torch.no_grad():
    obs, rew, dones, extras = env.step(pol(obs))
  if i % every == 0 or bool((tm.terminated & alive).any()):
    sp = ((robot.data.root_link_pos_w[:, :2] - st.center) * st.u).sum(-1) - span
    fp = robot.data.body_link_pos_w[:, fids, :]
    fs = ((fp[..., :2] - st.center[:, None]) * st.u[:, None]).sum(-1) - span[:, None]
    fz = fp[..., 2] - u.scene.env_origins[:, None, 2]
    z = robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]
    vf = (robot.data.root_link_lin_vel_w[:, :2] * st.face).sum(-1)
    q = robot.data.joint_pos[:, kn]
    tq = robot.data.qfrc_actuator[:, kn]
    gx = robot.data.projected_gravity_b[:, 0]
    on = (~st.arrived & ~st.paused & (u.episode_length_buf >= st.cmd_on))
    for e in range(NST):
      if alive[e]:
        log[e].append((i * u.step_dt, float(sp[e]), float(z[e]), float(gx[e]), float(vf[e]), fs[e].tolist(), fz[e].tolist(),
                       st.load[e].tolist(), q[e].tolist(), tq[e].tolist(), int(on[e]), int(st.paused[e]), int(st.arrived[e])))
  alive &= ~tm.terminated
def show(e, tail):
  r = log[e]; t_end = r[-1][0]
  for x in r:
    if x[0] < t_end - tail: continue
    print(f" {x[0]:5.1f} s {x[1]:+.2f} z {x[2]:.3f} gx {x[3]:+.2f} v {x[4]:+.2f} | fs {x[5][0]:+.2f} {x[5][1]:+.2f} fz {x[6][0]:.3f} {x[6][1]:.3f} load {x[7][0]:.2f} {x[7][1]:.2f} | knee {math.degrees(x[8][0]):5.1f} {math.degrees(x[8][1]):5.1f} tq {x[9][0]:+5.0f} {x[9][1]:+5.0f} | on {x[10]} pause {x[11]} arr {x[12]}")
import math
fallen = [e for e in range(NST) if not bool(alive[e])]
print("FALLEN", fallen, "of", NST, "| flat falls", int((~alive[NST:]).sum()), "of", N - NST)
for e in fallen[:int(os.environ.get("NSHOW", "6"))]:
  print(f"== env {e} lvl {int(lv[e])} {'monte' if want_climb[e] else 'desc'} falls at {log[e][-1][0]:.1f}")
  show(e, float(os.environ.get("TAIL", "3")))
ok = [e for e in range(NST) if bool(alive[e])]
if ok:
  print(f"== survivor env {ok[0]} lvl {int(lv[ok[0]])} {'monte' if want_climb[ok[0]] else 'desc'} (first 8 s)")
  r = log[ok[0]]; log[ok[0]] = [x for x in r if x[0] <= 8]; show(ok[0], 99)
