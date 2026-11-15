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
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev))
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
tm = u.termination_manager
rm = u.reward_manager
alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
acc = torch.zeros(len(rm._term_names), device=u.device); n = 0.0
accf = torch.zeros(len(rm._term_names), device=u.device); nf = 0.0
STOCH = os.environ.get("STOCH", "0") == "1"
fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
arrived = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
v = torch.zeros(u.num_envs, device=u.device); single = torch.zeros(u.num_envs, device=u.device)
robot = u.scene["robot"]
for i in range(steps):
  with torch.no_grad():
    obs, rew, dones, extras = env.step(pol(obs, stochastic_output=STOCH))
  m = st.is_stairs & alive
  acc += (rm._step_reward * m[:, None]).sum(0); n += float(m.sum())
  mf = ~st.is_stairs & alive
  accf += (rm._step_reward * mf[:, None]).sum(0); nf += float(mf.sum())
  v += (robot.data.root_link_lin_vel_w[:, :2] * st.u).sum(-1).abs() * m
  pos, valid, loaded = ST._support(u)
  single += ((loaded.sum(-1) == 1) & m).float()
  arrived |= st.arrived & alive
  fell |= tm.terminated & alive
  alive &= ~tm.terminated
s = st.is_stairs
print(f"{label}: stair envs {int(s.sum())} arrived {int(arrived[s].sum())} fell {int(fell[s].sum())} | |v| {float(v[s].sum() / (n * 1.0)):.3f} m/s single-support {100 * float(single[s].sum()) / n:.0f}%")
print("  per-term rate (weighted, per s):", " ".join(f"{k}={float(acc[j] / n):+.2f}" for j, k in enumerate(rm._term_names) if abs(float(acc[j] / n)) > 0.02))
print(f"  net reward rate per s: stair envs {float(acc.sum() / max(n, 1)):+.2f} | flat envs {float(accf.sum() / max(nf, 1)):+.2f} (stochastic={STOCH})")
print("  flat per-term:", " ".join(f"{k}={float(accf[j] / max(nf, 1)):+.2f}" for j, k in enumerate(rm._term_names) if abs(float(accf[j] / max(nf, 1))) > 0.05))
