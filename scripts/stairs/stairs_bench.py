"""stairs_bench.py <run> <ckpt> [label]: deterministic stairs bench, one line per height.

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
if os.environ.get("STOCH", "0") == "1":  # sampled actions, as in training
  _actor = runner.alg.actor
  pol = lambda o: _actor(o, stochastic_output=True)
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
alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
fell_t = torch.full((u.num_envs,), -1.0, device=u.device)
arrived = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
kick = torch.zeros(u.num_envs, device=u.device)
legc = torch.zeros(u.num_envs, device=u.device)
cp_out = torch.zeros(u.num_envs, device=u.device)
stance = torch.zeros(u.num_envs, device=u.device)
static = torch.zeros(u.num_envs, device=u.device)
single = torch.zeros(u.num_envs, device=u.device)
prog = torch.zeros(u.num_envs, device=u.device)
rm = u.reward_manager
_ib = rm._term_names.index("motion_body_pos") if "motion_body_pos" in rm._term_names else None
_wb = rm._term_cfgs[_ib].weight if _ib is not None else 1.0
trk = torch.zeros(u.num_envs, device=u.device)
trk_n = torch.zeros(u.num_envs, device=u.device)
for i in range(steps):
  with torch.no_grad():
    if os.environ.get("BENCH_FLAT_SCAN"):
      # Hide the steps: every scan ray reads the env's median height.
      a = obs["actor"]
      a[:, 126:] = a[:, 126:].median(-1, keepdim=True).values
    obs, rew, dones, extras = env.step(pol(obs))
  tm = u.termination_manager
  newly = tm.terminated & alive
  fell_t[newly] = i * u.step_dt
  alive &= ~tm.terminated
  arrived |= st.arrived & alive
  prog = torch.where(alive, torch.maximum(prog, st.progress), prog)
  if _ib is not None: trk += rm._step_reward[:, _ib] / _wb * alive
  trk_n += alive.float()
  kick += ST.stairs_riser_kick(u) * alive * u.step_dt
  legc += (ST.stairs_leg_contact(u) > 0) * alive * u.step_dt
  pos, valid, loaded = ST._support(u)
  standing = st.arrived | (u.episode_length_buf < st.cmd_on)
  armed = u.episode_length_buf >= st.hold_end
  ds = (loaded.all(-1) | (standing & loaded.any(-1))) & valid.any(-1) & alive & armed
  ss = (loaded.sum(-1) == 1) & alive & armed
  cp_out += (ST.stairs_capture_point(u) > 0) * ds
  stance += ds.float()
  static += ST.stairs_static_balance(u) * ss
  single += ss.float()

out = [f"STAIRS {label}:"]
for lvl in LEVELS:
  h = ST.H_RANGE[0] + lvl / max(ST.ROWS - 1, 1) * (ST.H_RANGE[1] - ST.H_RANGE[0])
  parts = []
  for climb, name in ((True, "monte"), (False, "descend")):
    m = st.is_stairs & (lv == lvl) & (want_climb == climb)
    n = int(m.sum())
    if n == 0:
      continue
    parts.append(f"{name} {int(arrived[m].sum())}/{n} tenu {int((arrived[m] & (fell_t[m] < 0)).sum())} chute {int((fell_t[m] >= 0).sum())} prog {float(prog[m].mean()):.2f}")
  m = st.is_stairs & (lv == lvl)
  parts.append(f"coups {kick[m].mean():.2f} BW.s tibia {legc[m].mean():.2f} s "
               f"CP hors (double appui) {100 * cp_out[m].sum() / stance[m].sum().clamp(min=1):.0f}% statique (appui simple) {100 * static[m].sum() / single[m].sum().clamp(min=1):.0f}%")
  out.append(f"  h {100 * h:.1f} cm: " + " | ".join(parts))
f = ~st.is_stairs
out.append(f"  plat: chutes {int((fell_t[f] >= 0).sum())}/{int(f.sum())} suivi body_pos {float(trk[f].sum() / trk_n[f].sum().clamp(min=1)):.3f}")
print("\n".join(out))
