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
from mjlab.tasks.tracking.config.rhps1 import vision_map as VM  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply  # noqa: E402

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
  N, dt, dv = u.num_envs, u.step_dt, u.device
  # Image failures (SCEN), applied to the frame the student reads, every EV_T s for EV_D s, all envs at once (their
  # episodes are out of phase): out = no return at all, freeze = the last frame held, person = a near vertical
  # band crossing the image (someone walking past the camera). The terrain is unchanged: the right answer is to
  # keep walking or, with GATE=1, to pause.
  SCEN = os.environ.get("SCEN", "none")
  EV_T, EV_D = float(os.environ.get("EV_T", "8")), float(os.environ.get("EV_D", "3"))
  # GATE=1, the controller-side prototype: an obstructed, blank or frozen frame pauses the stairs command (the
  # training's mid-walk pause, ARRIVE_LOADED waits for both feet down), resumed GATE_HOLD s after the frame is
  # clear again. Obstructed: more than GATE_FRAC of the pixels closer than GATE_NEAR m, or more than GATE_HFRAC
  # of them higher than GATE_HIGH m above the lowest sole (a person; a stair never is). Blank: less than GATE_VALID of
  # the pixels return (those outside the WFOV circle never do). Frozen: frame unchanged for GATE_FROZEN s (on the robot: its timestamp).
  GATE = os.environ.get("GATE", "0") == "1"
  G_NEAR, G_FRAC = float(os.environ.get("GATE_NEAR", "0.6")), float(os.environ.get("GATE_FRAC", "0.15"))
  G_VALID, G_FROZEN = float(os.environ.get("GATE_VALID", "0.1")), float(os.environ.get("GATE_FROZEN", "0.3"))
  G_HIGH, G_HFRAC = float(os.environ.get("GATE_HIGH", "0.6")), float(os.environ.get("GATE_HFRAC", "0.15"))
  G_HOLD = int(float(os.environ.get("GATE_HOLD", "1.0")) / dt)
  G_MASK = os.environ.get("GATE_MASK", "0") == "1"
  r_off, r_dir = D.HeadDepthPatternCfg().generate_rays(None, dv)
  cam = u.scene["depth_cam"]
  robot = u.scene["robot"]
  feet = [list(robot.body_names).index(k) for k in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
  high_max = torch.zeros(N, device=dv)
  high_ev = torch.zeros(N, device=dv)
  since_ev = 10**6
  falls_ev = 0
  held = None
  same = torch.zeros(N, dtype=torch.long, device=dv)
  clear_t = torch.full((N,), 10**6, dtype=torch.long, device=dv)
  ev_steps = gate_steps = fp_steps = 0
  near_max = torch.zeros(N, device=dv)
  valid_min = torch.ones(N, device=dv)
  # PERSON_W "lo,hi": band width in px (default 8,31); PERSON_EDGE=1: the band grazes one edge instead of crossing.
  P_W = tuple(int(x) for x in os.environ.get("PERSON_W", "8,31").split(","))
  P_EDGE = os.environ.get("PERSON_EDGE", "0") == "1"
  for i in range(steps):
    img = obs["depth"]
    t = i * dt
    in_ev = SCEN != "none" and t >= EV_T - EV_D and (t % EV_T) >= EV_T - EV_D
    if in_ev:
      ev_steps += 1
      if SCEN == "out":
        img = torch.zeros_like(img)
      elif SCEN == "freeze":
        held = img.clone() if held is None else held
        img = held
      elif SCEN == "person":
        ph = ((t % EV_T) - (EV_T - EV_D)) / EV_D  # 0 -> 1 across the event
        g = torch.Generator().manual_seed(int(t // EV_T))
        wpx = int(P_W[0] + torch.randint(0, P_W[1] - P_W[0] + 1, (1,), generator=g))
        d = float(0.4 + 1.1 * torch.rand(1, generator=g))
        x0 = int(-wpx + ph * (D.W + wpx))
        if P_EDGE:  # grazing: the band stays on one image edge for the whole event
          x0 = 0 if int(t // EV_T) % 2 == 0 else D.W - wpx
        im = img.view(N, D.H, D.W).clone()
        im[:, :, max(x0, 0):max(min(x0 + wpx, D.W), 0)] = d / D.RANGE[1]
        img = im.view(N, -1)
    else:
      held = None
    since_ev = 0 if in_ev else since_ev + 1
    valid = img > 0
    rng = (img * D.RANGE[1]).unsqueeze(-1)
    pts = cam.data.pos_w[:, None] + quat_apply(cam.data.quat_w[:, None].expand(-1, rng.shape[1], -1),
                                               r_off[None] + r_dir[None] * rng)
    sole = robot.data.body_link_pos_w[:, feet, 2].min(-1).values - VM.SOLE_DEPTH
    high = (valid & (pts[..., 2] - sole[:, None] > G_HIGH)).float().mean(-1)
    if in_ev:
      high_ev = torch.maximum(high_ev, high)
    else:
      high_max = torch.maximum(high_max, high)
    if GATE:
      near = (valid & (img < G_NEAR / D.RANGE[1])).float().mean(-1)
      if not in_ev:
        near_max = torch.maximum(near_max, near)
      vf = valid.float().mean(-1)
      blank = vf < G_VALID
      if not in_ev:
        valid_min = torch.minimum(valid_min, vf)
      # Frozen from the stream itself, as the robot's frame timestamp: a still robot on a clean frame repeats it.
      same = same + 1 if (in_ev and SCEN == "freeze") else torch.zeros_like(same)
      bad = (near > G_FRAC) | (high > G_HFRAC) | blank | (same * dt >= G_FROZEN)
      clear_t = torch.where(bad, torch.zeros_like(clear_t), clear_t + 1)
      gated = st.is_stairs & (clear_t < G_HOLD)
      if G_MASK:  # the controller also hands the policy a blank frame (no return), harmless on the bench
        img = torch.where(gated[:, None], torch.zeros_like(img), img)
      ep = u.episode_length_buf
      in_win = (ep >= st.pause0) & (ep < st.pause1)
      st.pause0 = torch.where(gated & ~in_win, ep, st.pause0)
      st.pause1 = torch.where(gated, torch.where(in_win, torch.maximum(st.pause1, ep + 2), ep + 2), st.pause1)
      gate_steps += int(gated.sum())
      if not in_ev:
        fp_steps += int(gated.sum())
    with torch.no_grad():
      if m is None:
        a = teacher(obs)
      else:
        a, h = m.step(obs["actor"][:, :n_prop], obs["actor"][:, n_prop:] if m.scan_dim else img, h)
      obs, _, dones, _ = env.step(a)
      if m is not None:
        h = h * (1 - dones.float())[None, :, None]
    arr += int(st.arrived_now.sum())
    if "stairs_fall" in u.termination_manager.active_terms:
      nf = int(u.termination_manager.get_term("stairs_fall").sum())
      falls += nf
      falls_ev += nf if since_ev * dt < 1.0 else 0
  qh = torch.quantile(high_max, torch.tensor([0.5, 0.99, 1.0], device=dv)).tolist()
  extra = (f" scen {SCEN} falls in events or 1 s after {falls_ev}; high frac (>{G_HIGH} m) outside events p50"
           f" {qh[0]:.3f} p99 {qh[1]:.3f} max {qh[2]:.3f}, in events median {float(high_ev.median()):.3f}")
  if GATE:
    ns = max(steps - ev_steps, 1) * N
    q = torch.quantile(near_max, torch.tensor([0.5, 0.9, 1.0], device=dv)).tolist()
    extra += (f" gate on {gate_steps / (steps * N):.3f} of env-steps, outside events {fp_steps / ns:.3f};"
              f" near frac max per env outside events p50 {q[0]:.3f} p90 {q[1]:.3f} max {q[2]:.3f}; valid frac min p0 {float(valid_min.min()):.3f}"
              f" p10 {float(torch.quantile(valid_min, 0.1)):.3f}")
  print(f"DEPTHEVAL {label}: arrivals {arr} falls {falls} over {u.num_envs} envs x {steps * u.step_dt:.0f} s{extra}")
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
