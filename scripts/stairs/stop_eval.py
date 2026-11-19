"""stop_eval.py <run> <ckpt> [label]: walk-then-stop bench on flat ground (Leo's gamepad test, tardis' stop bench).

Flat tiles, forward command for WALK_S s, then 0 for STOP_S s (a pause that never ends), deterministic policy.
  walk (from 3 s to the release): pelvis and chest roll p95 (deg), lateral pelvis oscillation peak-to-peak (cm)
  stop (after the release): slip of the loaded soles (horizontal travel while carrying > 25 % BW, cm, both feet),
  landings (corrective steps), time until still (pelvis < 0.02 m/s and |yaw rate| < 0.05 rad/s until the end, s),
  feet stagger along the heading at the end (|dx|, cm), roll max (deg)
  BENCH_N=32 WALK_S=8 STOP_S=5 python stop_eval.py <run> <ckpt> [label]
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
for k in ("RHPS1_STAIRS_FRAC", "RHPS1_STAIRS_STAND_P", "RHPS1_STAIRS_PAUSE_P", "RHPS1_STAIRS_STAIR_PAUSE_P",
          "RHPS1_STAIRS_OMNI_P"):
  os.environ[k] = "0"
os.environ["RHPS1_STAIRS_GAP"] = "2.5,2.5"  # far goal: the release comes before the arrival
os.environ["RHPS1_STAIRS_YAW_SPAWN"] = "0"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ck
N = int(os.environ.get("BENCH_N", "32"))
WALK, STOP = float(os.environ.get("WALK_S", "8")), float(os.environ.get("STOP_S", "5"))
dev = os.environ.get("BENCH_DEV", "cuda:0")
torch.manual_seed(int(os.environ.get("BENCH_SEED", "0")))

cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
u = env.unwrapped
st = u._stairs
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
p = Path(run) / ck if Path(run).is_absolute() else Path("logs/rsl_rl/rhps1_tracking") / run / ck
runner.load(str(p), load_cfg={"actor": True}, strict=True, map_location=dev)
pol = runner.get_inference_policy(device=dev)
u.reset()
obs = env.get_observations()
robot = u.scene["robot"]
bn = list(robot.body_names)
i_body, i_chest = bn.index("BODY"), bn.index("CHEST_P_LINK")
fids = st.foot_ids
BW = 57.64 * 9.81
dt = u.step_dt
rel = st.cmd_on + int(WALK / dt)  # release step per env
st.pause0[:] = rel
st.pause1[:] = 10**9


def roll(q):
  w, x, y, z = q.unbind(-1)
  return torch.rad2deg(torch.asin((2 * (y * z + w * x)).clamp(-1, 1)))


alive = torch.ones(N, dtype=torch.bool)
fell = torch.zeros(N, dtype=torch.bool)
walk_pr, walk_cr, walk_lat = [[] for _ in range(N)], [[] for _ in range(N)], [[] for _ in range(N)]
stop_roll = torch.zeros(N)
slip = torch.zeros(N)
lands = torch.zeros(N)
last_move = torch.zeros(N)
prev_f = None
prev_lo = torch.zeros(N, 2, dtype=torch.bool)
anchor = torch.zeros(N, 2, 2)
p_start = robot.data.root_link_pos_w[:, :2].clone()
steps = int((WALK + STOP + 2.0) / dt)
for _ in range(steps):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  term = u.termination_manager.terminated.cpu()
  fell |= term & alive
  alive &= ~term
  ep = u.episode_length_buf
  walking = ((ep >= st.cmd_on + int(3.0 / dt)) & (ep < rel)).cpu() & alive
  stopped = (ep >= rel).cpu() & alive
  qb = robot.data.body_link_quat_w
  pr, cr = roll(qb[:, i_body]).cpu(), roll(qb[:, i_chest]).cpu()
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  hf = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
  pp = robot.data.root_link_pos_w[:, :2] - p_start
  lat = (pp[:, 0] * -hf[:, 1] + pp[:, 1] * hf[:, 0]).cpu()
  fpos = robot.data.body_link_pos_w[:, fids, :2]
  loaded = (st.load > 0.25 * BW).cpu()
  # net travel of each loaded sole per stance after the release (a path length summed the 50 Hz jitter)
  lo = (st.load > 0.25 * BW).cpu() & stopped[:, None]
  fc = fpos.cpu()
  if prev_f is not None:
    endc = prev_lo & ~lo
    slip += ((fc - anchor).norm(dim=-1) * endc).sum(-1)
    anchor = torch.where((lo & ~prev_lo)[..., None], fc, anchor)
  else:
    anchor = fc.clone()
  prev_lo = lo
  prev_f = fc
  lands += (st.land.any(-1).cpu() & stopped).float()
  v = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1).cpu()
  wz = robot.data.root_link_ang_vel_w[:, 2].abs().cpu()
  moving = stopped & ((v > 0.02) | (wz > 0.05))
  last_move = torch.where(moving, ((ep - rel).float() * dt).cpu(), last_move)
  stop_roll = torch.where(stopped, torch.maximum(stop_roll, pr.abs()), stop_roll)
  for e in walking.nonzero().flatten().tolist():
    walk_pr[e].append(float(pr[e]))
    walk_cr[e].append(float(cr[e]))
    walk_lat[e].append(float(lat[e]))
slip += ((prev_f - anchor).norm(dim=-1) * prev_lo).sum(-1)
d = fpos[:, 0] - fpos[:, 1]
stagger = (d * hf).sum(-1).abs().cpu() * 100

ok = (~fell).nonzero().flatten().tolist()


def p95(lists):
  a = np.abs(np.concatenate([np.array(x) for x in lists if x] or [np.zeros(1)]))
  return float(np.percentile(a, 95))


def p2p(x):
  # lateral oscillation: detrended (linear fit) peak-to-peak
  if len(x) < 10:
    return float("nan")
  a = np.array(x)
  t = np.arange(len(a))
  a = a - np.polyval(np.polyfit(t, a, 1), t)
  return float(a.max() - a.min()) * 100


print(f"STOP {label}: {len(ok)}/{N} sans chute")
print(f"  marche: roulis bassin p95 {p95([walk_pr[e] for e in ok]):.1f} buste p95 {p95([walk_cr[e] for e in ok]):.1f} deg"
      f" | oscillation laterale p2p med {np.nanmedian([p2p(walk_lat[e]) for e in ok]):.1f} cm")
print(f"  arret: glissement pieds charges med {float(slip[ok].median()) * 100:.1f} p90 {float(slip[ok].quantile(0.9)) * 100:.1f} cm"
      f" | poses apres arret med {float(lands[ok].median()):.0f}"
      f" | immobile apres med {float(last_move[ok].median()):.1f} p90 {float(last_move[ok].quantile(0.9)):.1f} s"
      f" | decalage pieds fin med {float(stagger[ok].median()):.1f} p90 {float(stagger[ok].quantile(0.9)):.1f} cm"
      f" | roulis max med {float(stop_roll[ok].median()):.1f} deg")
