"""stop_eval.py <run> <ckpt> [label]: walk-then-stop bench on flat ground (Leo's gamepad test, tardis' stop bench).

Flat tiles, forward command for WALK_S s, then 0 for STOP_S s (a pause that never ends), deterministic policy.
  walk (from 3 s to the release): pelvis and chest roll p95 (deg), lateral pelvis oscillation peak-to-peak (cm)
  stop (after the release): slip of the loaded soles (horizontal travel while carrying > 25 % BW, cm, both feet),
  landings (corrective steps), time until still (pelvis < 0.02 m/s and |yaw rate| < 0.05 rad/s until the end, s),
  feet stagger along the heading at the end (|dx|, cm), roll max (deg)
  BENCH_N=32 WALK_S=8 STOP_S=5 python stop_eval.py <run> <ckpt> [label]
"""

import math
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
os.environ["RHPS1_STAIRS_OSEQ_P"] = "0"  # forward-then-stop on every env (recipes with OSEQ_P left random sequences here)
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
assert "L" in bn[fids[0]][:2] and "R" in bn[fids[1]][:2], [bn[i] for i in fids]
BW = 57.64 * 9.81
dt = u.step_dt
rel = st.cmd_on + int(WALK / dt)  # release step per env
st.pause0[:] = rel
st.pause1[:] = 10**9


def roll(q):
  w, x, y, z = q.unbind(-1)
  return torch.rad2deg(torch.asin((2 * (y * z + w * x)).clamp(-1, 1)))


def pitch(q):
  w, x, y, z = q.unbind(-1)
  return torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))


# rest posture (tardis 2026-11-26: backward falls on the robot from a crouched rest, pelvis tilted back): over the
# last 2 s of the stop, pelvis pitch vs its value at the reset pose q0, and leg q - q0
jn = list(robot.joint_names)
i_leg = [i for i, n in enumerate(jn) if any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]
q0 = robot.data.default_joint_pos.clone()
pitch0 = None
rest_p, rest_n = torch.zeros(N), torch.zeros(N)
rest_c = torch.zeros(N)  # CoM ahead of the ankle midpoint along the heading, m
rest_dq = torch.zeros(N, len(jn))


alive = torch.ones(N, dtype=torch.bool)
fell = torch.zeros(N, dtype=torch.bool)
walk_pr, walk_cr, walk_lat = [[] for _ in range(N)], [[] for _ in range(N)], [[] for _ in range(N)]
stop_roll = torch.zeros(N)
slip = torch.zeros(N)
lands = torch.zeros(N)
vland, fpeak, trq_r = [], [], []  # touchdown sole vz (m/s), 60 ms peak sole load (N), leg |torque| / limit
knee_t = []  # |knee actuator torque| (N.m), L and R, while walking
kid_t = [list(robot.joint_names).index(n) for n in ("L_KNEE_P", "R_KNEE_P")]
wgaps = [[] for _ in range(N)]
single_t = torch.zeros(N)
walk_t = torch.zeros(N)
hike = torch.zeros(N)
hike_n = torch.zeros(N)
lift_pk = torch.zeros(N, 2)
lifts = [[] for _ in range(N)]
knee_v = []
kq_pk = torch.zeros(N, 2)
kq_st = torch.zeros(N, 2)
kq_n = torch.zeros(N, 2)
kexc = [[] for _ in range(N)]
jk = [list(robot.joint_names).index(n) for n in ('L_KNEE_P', 'R_KNEE_P')]
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
  pp = pitch(qb[:, i_body]).cpu()
  if pitch0 is None:
    pitch0 = pp.clone()
  rest = (ep >= rel + int((STOP - 2.0) / dt)).cpu() & alive
  rest_p += pp * rest
  rest_dq += (robot.data.joint_pos - q0).cpu() * rest[:, None]
  rest_n += rest.float()
  fp = robot.data.body_link_pos_w[:, st.foot_ids, :2]
  qr = robot.data.root_link_quat_w
  yw = torch.atan2(2 * (qr[:, 0] * qr[:, 3] + qr[:, 1] * qr[:, 2]), 1 - 2 * (qr[:, 2] ** 2 + qr[:, 3] ** 2))
  cm = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id, :2] - fp.mean(1)
  rest_c += (cm[:, 0] * torch.cos(yw) + cm[:, 1] * torch.sin(yw)).cpu() * rest
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
  wl = st.land.any(-1).cpu() & walking
  lw = st.land.cpu() & walking[:, None]
  vland.append(st.v_land.cpu()[lw])
  fw = st.f_close.cpu() & walking[:, None]
  fpeak.append(st.f_peak.cpu()[fw])
  knee_t.append(robot.data.qfrc_actuator[:, kid_t].abs().cpu()[walking])
  lim = getattr(u, "_st_trq_lim", None)
  if lim is not None and walking.any():
    legs = torch.isfinite(lim)
    trq_r.append((robot.data.qfrc_actuator.abs() / lim)[:, legs].cpu()[walking].flatten())
  for e in wl.nonzero().flatten().tolist():
    wgaps[e].append(float(st.land_gap[e]))
  single_t += ((st.load > 0.1 * BW).sum(-1).cpu() == 1).float() * walking
  walk_t += walking.float()
  # pelvis roll toward the swing side in single support: > 0 = swing side up (hip hike), < 0 = it drops
  ldd = (st.load > 0.1 * BW).cpu()
  one = (ldd.sum(-1) == 1) & walking
  sgn = torch.where(ldd[:, 1], 1.0, -1.0)  # right foot loaded -> left swing; +roll lifts the left side
  hike += torch.where(one, pr * sgn, torch.zeros_like(pr))
  hike_n += one.float()
  # swing sole height (ankle - 0.10 m over the ground under it): peak per swing, recorded at the landing
  hgt = (robot.data.body_link_pos_w[:, fids, 2] - 0.10 - u.scene["stairs_ground"].data.hit_pos_w[:, :2, 2]).cpu()
  sw = (st.load <= 10.0).cpu()
  lift_pk = torch.where(sw, torch.maximum(lift_pk, hgt), lift_pk)
  ld = st.land.cpu() & walking[:, None]
  for e, k in ld.nonzero().tolist():
    lifts[e].append(float(lift_pk[e, k]))
  lift_pk = torch.where(sw, lift_pk, torch.zeros_like(lift_pk))
  kq = robot.data.joint_pos[:, jk].cpu()
  kq_pk = torch.where(sw, torch.maximum(kq_pk, kq), kq_pk)
  stn = ~sw & walking[:, None]
  kq_st += kq * stn
  kq_n += stn.float()
  for e, k in ld.nonzero().tolist():
    if kq_n[e, k] > 0:
      kexc[e].append(float(kq_pk[e, k] - kq_st[e, k] / kq_n[e, k]))
  kq_pk = torch.where(sw, kq_pk, kq)
  kv = robot.data.joint_vel[:, jk].abs().cpu()
  knee_v.append(kv[walking].flatten())
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
gp = [g for e in ok for g in wgaps[e] if g < 5]
print(f"  cadence: pose a pose med {np.median(gp) if gp else float('nan'):.2f} s | simple appui"
      f" {float((single_t[ok] / walk_t[ok].clamp(min=1)).mean()) * 100:.0f} % du temps de marche"
      f" | roulis vers le pied en l'air moy {float((hike[ok] / hike_n[ok].clamp(min=1)).mean()):+.1f} deg")
lf = [x for e in ok for x in lifts[e]]
kvv = torch.cat(knee_v)
print(f"  pied: lever max par pas med {np.median(lf) * 100 if lf else float('nan'):.1f} cm"
      f" | genou balancier - appui med {np.degrees(np.median([x for e in ok for x in kexc[e]] or [np.nan])):+.0f} deg"
      f" | genou |v| p99 {float(kvv.quantile(0.99)) if len(kvv) else float('nan'):.2f} rad/s (plafond 2,64)")
vl, fp = torch.cat(vland), torch.cat(fpeak)
tr = torch.cat(trq_r) if trq_r else torch.zeros(1)
q = lambda x, a: float(x.quantile(a)) if len(x) else float("nan")  # noqa: E731
print(f"  impacts: vz pose p50/p90 {q(vl, 0.5):.2f}/{q(vl, 0.9):.2f} m/s | Fz pic 60 ms p50/p90 {q(fp, 0.5):.0f}/"
      f"{q(fp, 0.9):.0f} N ({q(fp, 0.9) / BW:.2f} BW) | couple jambes / limite p99 {q(tr, 0.99):.2f}, >= 0.9 "
      f"{100 * float((tr >= 0.9).float().mean()):.1f} %")
kt = torch.cat(knee_t) if knee_t else torch.zeros(1, 2)
print(f"  genou couple p99 G/D {q(kt[:, 0], 0.99):.0f}/{q(kt[:, 1], 0.99):.0f} N.m (reducteur 70) | >= 66.5 N.m (95 %) G/D "
      f"{100 * float((kt[:, 0] >= 66.5).float().mean()):.1f}/{100 * float((kt[:, 1] >= 66.5).float().mean()):.1f} %")
print(f"  arret: glissement pieds charges med {float(slip[ok].median()) * 100:.1f} p90 {float(slip[ok].quantile(0.9)) * 100:.1f} cm"
      f" | poses apres arret med {float(lands[ok].median()):.0f}"
      f" | immobile apres med {float(last_move[ok].median()):.1f} p90 {float(last_move[ok].quantile(0.9)):.1f} s"
      f" | decalage pieds fin med {float(stagger[ok].median()):.1f} p90 {float(stagger[ok].quantile(0.9)):.1f} cm"
      f" | roulis max med {float(stop_roll[ok].median()):.1f} deg")
rk = [e for e in ok if rest_n[e] > 0]
if rk:
  rp = (rest_p[rk] / rest_n[rk]).mean()
  dq = torch.rad2deg(rest_dq[rk] / rest_n[rk][:, None]).mean(0)
  g = {k: float(dq[[jn.index(f"{s}_{k}") for s in ("L", "R")]].mean()) for k in ("CROTCH_P", "KNEE_P", "ANKLE_P")}
  print(f"  repos: tangage bassin {float(rp):+.1f} deg (q0 {float(pitch0[rk].mean()):+.1f}, ecart {float(rp - pitch0[rk].mean()):+.1f})"
        f" | jambes q - q0 hanche P {g['CROTCH_P']:+.1f} genou {g['KNEE_P']:+.1f} cheville P {g['ANKLE_P']:+.1f} deg"
        f" | max |q - q0| jambes {float(dq[i_leg].abs().max()):.1f} deg")
  from mjlab.tasks.tracking.config.rhps1.stairs import STANCE_DEG

  sv = {k: float(dq[[jn.index(f"{s}_{k}") for s in ("L", "R")]].mean() + math.degrees(q0[0, jn.index(f"L_{k}")]) - STANCE_DEG[k])
        for k in ("CROTCH_P", "KNEE_P", "ANKLE_P")}
  print(f"  repos vs posture debout reelle (slot 18 y revient): hanche P {sv['CROTCH_P']:+.1f} genou {sv['KNEE_P']:+.1f}"
        f" cheville P {sv['ANKLE_P']:+.1f} deg | CdM devant les chevilles {100 * float((rest_c[rk] / rest_n[rk]).mean()):+.1f} cm"
        f" (slot 18 mc_mujoco +4.6)")
