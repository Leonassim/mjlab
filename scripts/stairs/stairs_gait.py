"""stairs_gait.py <run> <ckpt> [label]: Leo's stairs criteria on the pinned bench (deterministic).

Per height and mode: arrival / held / falls, trunk tilt (pelvis BODY and CHEST_P_LINK, deg; pitch + = leaning
forward), landings counted as in training (>= 80 ms airborne: speed in the 50 ms before, 60 ms peak load),
step pattern on the treads (over = passes the other foot, to = joins it), forward speed, actuator saturation.
  BENCH_LEVELS=0,2,4 BENCH_PER=4 python stairs_gait.py <run> <ckpt> [label]
"""
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ck
LEVELS = [int(x) for x in os.environ.get("BENCH_LEVELS", "0,2,4").split(",")]
PER = int(os.environ.get("BENCH_PER", "4"))
T = float(os.environ.get("BENCH_T", "24"))
dev = os.environ.get("BENCH_DEV", "cpu")
torch.manual_seed(int(os.environ.get("BENCH_SEED", "0")))

cells = [(lv, climb) for lv in LEVELS for climb in (True, False)]
N = len(cells) * PER
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = N


def _fall_all(env):
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
kinds = cfg.commands["motion"].stairs_kinds
col = kinds.index("up")
lv = torch.tensor([c[0] for c in cells for _ in range(PER)], device=u.device)
ty = torch.full((N,), col, device=u.device)
terrain.terrain_levels[:] = lv
terrain.terrain_types[:] = ty
terrain.env_origins[:] = terrain.terrain_origins[lv, ty]
st.kind = torch.ones(N, dtype=torch.long, device=u.device)
st.is_stairs = st.kind > 0
st.tread = torch.full((N,), cfg.commands["motion"].stairs_treads[col], device=u.device)
st.nsteps = torch.full((N,), ST.NSTEPS, device=u.device)
want = torch.tensor([c[1] for c in cells for _ in range(PER)], device=u.device)
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True}, strict=True, map_location=dev)
pol = runner.get_inference_policy(device=dev)
if os.environ.get("BENCH_STOCH"):
  _k = float(os.environ["BENCH_STOCH"])  # noise as a fraction of the trained std
  def pol(o, _a=runner.alg.actor):
    m = _a(o, stochastic_output=False)
    return m + _k * _a.distribution.std_param.abs() * torch.randn_like(m) if hasattr(_a, 'distribution') else m
u.reset()
cmd = u.command_manager.get_term("motion")
for _ in range(30):
  bad = (st.climb != want).nonzero().flatten()
  if len(bad) == 0:
    break
  cmd._in_reset = True
  cmd._resample_command(bad)
  cmd._in_reset = False
u.sim.forward()
obs = env.get_observations()

robot = u.scene["robot"]
bn = list(robot.body_names)
i_body, i_chest = bn.index("BODY"), bn.index("CHEST_P_LINK")
fids = st.foot_ids
mjm = u.sim.mj_model
lim = torch.tensor(np.abs(mjm.actuator_forcerange[:, 1]), dtype=torch.float32)
limited = torch.tensor(mjm.actuator_forcelimited.astype(bool)) & (lim > 0)
anames = [mjm.actuator(i).name for i in range(mjm.nu)]
groups = {g: [i for i, n in enumerate(anames) if g in n and limited[i]] for g in ("KNEE", "ANKLE_P", "ANKLE_R", "CROTCH_P", "CROTCH_R")}
span = ST.PLATFORM / 2 + (ST.NSTEPS - 1) * st.tread
dirn = torch.where(want, 1, -1)


def tilt(q):
  # world gravity in the body frame -> (pitch, roll) in deg, pitch + = leaning forward
  w, x, y, z = q.unbind(-1)
  gx = -(2 * (x * z - w * y))
  gy = -(2 * (y * z + w * x))
  return torch.rad2deg(torch.asin(gx.clamp(-1, 1))), torch.rad2deg(torch.asin(gy.clamp(-1, 1)))


def level(s_rel, t):
  return (-(s_rel) / t).ceil().clamp(0, ST.NSTEPS)


alive = torch.ones(N, dtype=torch.bool)
fell = torch.zeros(N, dtype=torch.bool)
arrived = torch.zeros(N, dtype=torch.bool)
rec = {k: [[] for _ in range(N)] for k in ("pp", "pr", "cp", "cr", "v", "lv", "lf", "ov", "per", "sw", "air")}
sat = {g: torch.zeros(N) for g in groups}
nact = torch.zeros(N)
foot_lvl = None
over = torch.zeros(N)
last_t = [None] * N
to = torch.zeros(N)
for i in range(int(T / u.step_dt)):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  term = u.termination_manager.terminated.cpu()
  fell |= term & alive
  alive &= ~term
  arrived |= st.arrived.cpu() & alive
  armed = ((u.episode_length_buf >= st.hold_end).cpu() & alive)
  qa = robot.data.body_link_quat_w
  pp, pr = tilt(qa[:, i_body])
  cp, cr = tilt(qa[:, i_chest])
  vf = (robot.data.root_link_lin_vel_w[:, :2] * st.face).sum(-1)
  s_rel = ((robot.data.body_link_pos_w[:, fids, :2] - st.center[:, None]) * st.u[:, None]).sum(-1) - span[:, None]
  # climb moves to negative s from outside; descend moves to positive s from the platform
  fl = level(s_rel, st.tread[:, None])
  if foot_lvl is None:
    foot_lvl = fl.clone()
  af = robot.data.actuator_force.abs().cpu()
  for g, ids in groups.items():
    if ids:
      sat[g] += ((af[:, ids] >= 0.97 * lim[ids]).any(-1) & armed).float()
  nact += armed.float()
  land = st.land.cpu()
  _d = robot.data.body_link_pos_w[:, fids[0], :2] - robot.data.body_link_pos_w[:, fids[1], :2]
  _q = robot.data.root_link_quat_w
  _yaw = torch.atan2(2 * (_q[:, 0] * _q[:, 3] + _q[:, 1] * _q[:, 2]), 1 - 2 * (_q[:, 2] ** 2 + _q[:, 3] ** 2))
  sw = (-_d[:, 0] * torch.sin(_yaw) + _d[:, 1] * torch.cos(_yaw)).abs().cpu()
  # Sole rays on a lower tread than the highest one under the foot: geometric, any riser height.
  rp = u.scene["stairs_soles"].data.hit_pos_w[..., :2].view(N, 2, -1, 2)
  rs = ((rp - st.center[:, None, None]) * st.u[:, None, None]).sum(-1) - span[:, None, None]
  rl = level(rs, st.tread[:, None, None])
  rays_off = (rl < rl.max(-1, keepdim=True).values).sum(-1).cpu()
  airmax = st.air.max(-1).values.cpu()
  for e in armed.nonzero().flatten().tolist():
    rec["pp"][e].append(float(pp[e])); rec["pr"][e].append(float(pr[e]))
    rec["cp"][e].append(float(cp[e])); rec["cr"][e].append(float(cr[e]))
    rec["v"][e].append(float(vf[e]))
    rec["sw"][e].append(float(sw[e]))
    rec["air"][e].append(float(airmax[e]))
    for k in (0, 1):
      if land[e, k]:
        if last_t[e] is not None and not bool(st.arrived[e]):
          rec["per"][e].append(i * u.step_dt - last_t[e])
        last_t[e] = i * u.step_dt
        rec["lv"][e].append(float(st.v_land[e, k]))
        rec["ov"][e].append(int(rays_off[e, k]))
        new, old, other = int(fl[e, k]), int(foot_lvl[e, k]), int(foot_lvl[e, 1 - k])
        if new != old:
          if dirn[e] * (new - other) > 0:
            over[e] += 1
          elif new == other:
            to[e] += 1
        foot_lvl[e, k] = fl[e, k]
      if st.f_close[e, k]:
        rec["lf"][e].append(float(st.f_peak[e, k]))


def cat(key, m):
  return np.array([x for e in m.nonzero().flatten().tolist() for x in rec[key][e]])


def pq(a, q):
  return float(np.percentile(a, q)) if len(a) else float("nan")


print(f"GAIT {label}:")
for lvl in LEVELS:
  h = ST.H_RANGE[0] + lvl / max(ST.ROWS - 1, 1) * (ST.H_RANGE[1] - ST.H_RANGE[0])
  for climb, name in ((True, "monte"), (False, "desc")):
    m = (lv.cpu() == lvl) & (want.cpu() == climb)
    pp_, pr_, cp_, cr_ = cat("pp", m), cat("pr", m), cat("cp", m), cat("cr", m)
    v_, lv_, lf_, ov_, per_, sw_ = cat("v", m), cat("lv", m), cat("lf", m), cat("ov", m), cat("per", m), cat("sw", m)
    s = {g: float(sat[g][m].sum() / nact[m].sum().clamp(min=1)) * 100 for g in groups}
    print(f"  h {100 * h:.1f} {name}: arr {int(arrived[m].sum())}/{int(m.sum())} tenu {int((arrived & ~fell)[m].sum())} chute {int(fell[m].sum())}"
          f" | bassin tangage moy {np.mean(pp_):+.1f} p95|.| {pq(np.abs(pp_), 95):.1f} roulis p95 {pq(np.abs(pr_), 95):.1f}"
          f" | buste tangage moy {np.mean(cp_):+.1f} p95 {pq(np.abs(cp_), 95):.1f} roulis p95 {pq(np.abs(cr_), 95):.1f} deg"
          f" | v moy {np.mean(v_):.2f} p95 {pq(v_, 95):.2f} m/s")
    print(f"      poses {len(lv_)}: v med {pq(lv_, 50):.2f} p90 {pq(lv_, 90):.2f} (>0.21 {100 * (lv_ > 0.21).mean() if len(lv_) else 0:.0f}%)"
          f" F med {pq(lf_, 50):.0f} p90 {pq(lf_, 90):.0f} N | marches: passe {int(over[m].sum())} rejoint {int(to[m].sum())}"
          f" | ecart pieds med {100 * pq(sw_, 50):.0f} p95 {100 * pq(sw_, 95):.0f} cm (init 20)"
          f" | periode pas med {pq(per_, 50):.2f} p10 {pq(per_, 10):.2f} s (<1 s {100 * (per_ < 1).mean() if len(per_) else 0:.0f}%)"
          f" | a cheval (>=3/6 rayons hors marche) {100 * (ov_ >= 3).mean() if len(ov_) else 0:.0f}% bord (>=1) {100 * (ov_ >= 1).mean() if len(ov_) else 0:.0f}%"
          f" | saturation % " + " ".join(f"{g} {s[g]:.0f}" for g in groups))
    air_ = cat("air", m)
    print(f"      jambe levee >1.2 s {100 * (air_ > 1.2).mean() if len(air_) else 0:.0f}% >2 s {100 * (air_ > 2.0).mean() if len(air_) else 0:.0f}%"
          f" du temps, plus long pied en l'air {air_.max() if len(air_) else 0:.1f} s")
