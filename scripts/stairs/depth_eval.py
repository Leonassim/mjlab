"""depth_eval.py: stairs rollouts of the depth-image student alone (or of the teacher, for reference).

Default (vismoke protocol, comparable with tardis's numbers): the training env as is, with resets, SMOKE_N envs
x SMOKE_T s, STAIRS_FRAC from the recipe (RHPS1_STAIRS_STAIRS_FRAC=1.0 for stairs only); counts arrivals
(st.arrived_now) and stairs_fall terminations. Deterministic actions.

PINNED=1, same pinning as kneestat.py (severe, its flat envs all fall on the copy): each stair env gets a fixed level and mode (climb / descend), PER per cell, plus
flat envs. No resets: a real fall (pelvis low, tilt, body contact) ends the env. Deterministic actions. The
depth image keeps its training degradations unless RHPS1_DEPTH_CLEAN=1.

  depth_eval.sh <student.pt | teacher> [label]     (RHPS1_DEPTH=1 RHPS1_HEAD_HOLD=1 + recipe, CPU by default)
  BENCH_LEVELS=2,4,6 BENCH_PER=8 BENCH_T=24
"""

import os
import sys
from dataclasses import asdict

import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import depth_student as D  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
TEACHER = os.environ.get("TEACHER", "logs/rsl_rl/rhps1_tracking/2026-11-15_10-23-12/model_15000.pt")
which = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(which)
LEVELS = [int(x) for x in os.environ.get("BENCH_LEVELS", "2,4,6").split(",")]
PER = int(os.environ.get("BENCH_PER", "8"))
NFLAT = int(os.environ.get("BENCH_FLAT", "16"))
T = float(os.environ.get("BENCH_T", "24"))
dev = os.environ.get("BENCH_DEV", "cpu")
torch.manual_seed(int(os.environ.get("BENCH_SEED", "0")))

PINNED = os.environ.get("PINNED", "0") == "1"
if not PINNED:
  cfg = load_env_cfg(TASK)
  cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
  cfg.scene.num_envs = int(os.environ.get("SMOKE_N", "32"))
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
  u = env.unwrapped
  st = u._stairs
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
  runner.load(TEACHER, load_cfg={"actor": True}, strict=True, map_location=dev)
  teacher = runner.get_inference_policy(device=dev)
  obs = env.get_observations()
  n_prop = obs["actor"].shape[-1] - ST.scan_dim()
  if which == "teacher":
    m = None
  else:
    ck = torch.load(which, map_location=dev)
    m = D.DepthStudent(ck["n_prop"], ck["n_act"], torch.zeros(ck["n_prop"]), torch.ones(ck["n_prop"]),
                     scan_dim=ck.get("scan_dim", 0)).to(dev)
    if any(k.startswith("scan_head") for k in ck["student"]):
      m.enable_scan_head(ST.scan_dim())
      m.to(dev)
    m.load_state_dict(ck["student"])
    m.eval()
    h = torch.zeros(1, u.num_envs, m.gru_dim, device=dev)
  arr = falls = 0
  steps = int(float(os.environ.get("SMOKE_T", "24")) / u.step_dt)
  for i in range(steps):
    with torch.no_grad():
      if m is None:
        a = teacher(obs)
      else:
        a, h = m.step(obs["actor"][:, :n_prop], obs["actor"][:, n_prop:] if m.scan_dim else obs["depth"], h)
      obs, _, dones, _ = env.step(a)
      if m is not None:
        h = h * (1 - dones.float())[None, :, None]
    arr += int(st.arrived_now.sum())
    if "stairs_fall" in u.termination_manager.active_terms:
      falls += int(u.termination_manager.get_term("stairs_fall").sum())
  print(f"DEPTHEVAL {label}: arrivals {arr} falls {falls} over {u.num_envs} envs x {steps * u.step_dt:.0f} s")
  sys.exit(0)

cells = [(lv, climb) for lv in LEVELS for climb in (True, False)]
NST = len(cells) * PER
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = NST + NFLAT


def _fall_all(env):
  robot = env.scene["robot"]
  ground = env.scene["stairs_ground"].data.hit_pos_w[:, 2, 2]
  low = robot.data.root_link_pos_w[:, 2] - ground < 0.5
  tilt = robot.data.projected_gravity_b[:, 2] > -0.55
  return low | tilt | (env.scene["stairs_body"].data.found.sum(-1) > 0)


cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
cfg.terminations["stairs_fall"].func = _fall_all
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
u = env.unwrapped
st = u._stairs
terrain = u.scene.terrain
kinds = cfg.commands["motion"].stairs_kinds
col_st, col_flat = kinds.index("up"), kinds.index("flat")
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
st.mode_climb = want_climb.clone()

runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(TEACHER, load_cfg={"actor": True}, strict=True, map_location=dev)
teacher = runner.get_inference_policy(device=dev)
u.reset()
cmd = u.command_manager.get_term("motion")
for _ in range(30):
  bad = (st.is_stairs & (st.climb != want_climb)).nonzero().flatten()
  if len(bad) == 0:
    break
  cmd._in_reset = True
  cmd._resample_command(bad)
  cmd._in_reset = False
u.sim.forward()
assert bool((st.climb[st.is_stairs] == want_climb[st.is_stairs]).all()), "mode pin failed (MODE_FIXED=0?)"
obs = env.get_observations()
n_prop = obs["actor"].shape[-1] - ST.scan_dim()

if which == "teacher":
  def act(o, h):
    return teacher(o), h
  h = None
else:
  ck = torch.load(which, map_location=dev)
  m = D.DepthStudent(ck["n_prop"], ck["n_act"], torch.zeros(ck["n_prop"]), torch.ones(ck["n_prop"]),
                     scan_dim=ck.get("scan_dim", 0)).to(dev)
  if any(k.startswith("scan_head") for k in ck["student"]):
    m.enable_scan_head(ST.scan_dim())
    m.to(dev)
  m.load_state_dict(ck["student"])
  m.eval()
  h = torch.zeros(1, u.num_envs, m.gru_dim, device=dev)

  def act(o, h):
    return m.step(o["actor"][:, :n_prop], o["actor"][:, n_prop:] if m.scan_dim else o["depth"], h)

N = u.num_envs
alive = torch.ones(N, dtype=torch.bool, device=u.device)
arrived = torch.zeros(N, dtype=torch.bool, device=u.device)
prog = torch.zeros(N, device=u.device)
steps = int(T / u.step_dt)
for i in range(steps):
  with torch.no_grad():
    a, h = act(obs, h)
    obs, _, _, _ = env.step(a)
  alive &= ~u.termination_manager.terminated
  arrived |= st.arrived & alive
  prog = torch.where(alive, torch.maximum(prog, st.progress), prog)

out = [f"DEPTHEVAL {label}:"]
tot_f = tot_a = tot_n = 0
for lvl in LEVELS:
  h_cm = 100 * (ST.H_RANGE[0] + lvl / max(ST.ROWS - 1, 1) * (ST.H_RANGE[1] - ST.H_RANGE[0]))
  parts = []
  for climb, name in ((True, "monte"), (False, "descend")):
    msk = st.is_stairs & (lv == lvl) & (want_climb == climb)
    n = int(msk.sum())
    f = int((~alive[msk]).sum())
    a = int((arrived[msk] & alive[msk]).sum())
    tot_f, tot_a, tot_n = tot_f + f, tot_a + a, tot_n + n
    parts.append(f"{name} arrive {a}/{n} chute {f} prog {float(prog[msk].mean()):.2f}")
  out.append(f"  h {h_cm:.1f} cm: " + " | ".join(parts))
fl = ~st.is_stairs
out.append(f"  plat: chutes {int((~alive[fl]).sum())}/{int(fl.sum())}")
out.append(f"  TOTAL escaliers: arrive {tot_a}/{tot_n} chute {tot_f}/{tot_n}")
print("\n".join(out))
