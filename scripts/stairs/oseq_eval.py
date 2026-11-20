"""oseq_eval.py <run> <ckpt> [label]: per-axis maneuvering bench on flat ground (o50, RHPS1_STAIRS_OSEQ_P).

Every env holds one command for MOVE_S s, then zero for STOP_S s, deterministic policy, no random pauses. Patterns (env
id mod 8): fwd, back, left, right, yaw+, yaw-, fwd+yaw+, fwd+left. Per pattern, from 1 s after the start:
  ratio: mean speed along each commanded axis / its target (fwd 0.12, back/side 0.08 m/s, yaw 0.3 rad/s)
  leak: mean |speed| on the uncommanded axes (m/s, rad/s for yaw), roll p95 (deg), min lateral ankle width (cm), falls
  stop: loaded-sole slip after the release (cm), landings after the release
  BENCH_N=64 MOVE_S=6 STOP_S=4 python oseq_eval.py <run> <ckpt> [label]
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
os.environ["RHPS1_STAIRS_OSEQ_P"] = "1"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1.stairs import OSEQ_V  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ck
N = int(os.environ.get("BENCH_N", "64"))
MOVE, STOP = float(os.environ.get("MOVE_S", "6")), float(os.environ.get("STOP_S", "4"))
dev = os.environ.get("BENCH_DEV", "cuda:0")
torch.manual_seed(int(os.environ.get("BENCH_SEED", "0")))
PAT = (("avant", (1, 0, 0)), ("arriere", (-1, 0, 0)), ("gauche", (0, 1, 0)), ("droite", (0, -1, 0)),
       ("lacet+", (0, 0, 1)), ("lacet-", (0, 0, -1)), ("avant+lacet", (1, 0, 1)), ("avant+gauche", (1, 1, 0)))

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
dt = u.step_dt
pid = torch.arange(N, device=dev) % len(PAT)
cmd = torch.tensor([c for _, c in PAT], dtype=torch.float, device=dev)[pid]
st.oseq[:] = True
st.seg_cmd[:] = 0.0
st.seg_cmd[:, 0] = cmd
st.seg_end[:] = 10**9
st.seg_end[:, 0] = int(MOVE / dt)
tgt = torch.stack([torch.where(cmd[:, 0] > 0, OSEQ_V[0], OSEQ_V[1]) * cmd[:, 0], OSEQ_V[2] * cmd[:, 1],
                   OSEQ_V[3] * cmd[:, 2]], 1).cpu()
robot = u.scene["robot"]
BW = 57.64 * 9.81
alive = torch.ones(N, dtype=torch.bool)
fell = torch.zeros(N, dtype=torch.bool)
vsum, vcnt = torch.zeros(N, 3), torch.zeros(N)
rolls = [[] for _ in range(N)]
slip, lands = torch.zeros(N), torch.zeros(N)
wmin = torch.full((N,), 9.0)
prev_f = None
prev_lo = torch.zeros(N, 2, dtype=torch.bool)
anchor = torch.zeros(N, 2, 2)
for _ in range(int((MOVE + STOP + 1.5) / dt)):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  term = u.termination_manager.terminated.cpu()
  fell |= term & alive
  alive &= ~term
  t = ((u.episode_length_buf - st.cmd_on).float() * dt).cpu()
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  hf = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
  v = robot.data.root_link_lin_vel_w[:, :2]
  vb = torch.stack([(v * hf).sum(-1), -v[:, 0] * hf[:, 1] + v[:, 1] * hf[:, 0],
                    robot.data.root_link_ang_vel_w[:, 2]], 1).cpu()
  mv = (t >= 1.0) & (t < MOVE) & alive
  vsum += vb * mv[:, None]
  vcnt += mv.float()
  w, x, y, z = q.unbind(-1)
  r = torch.rad2deg(torch.asin((2 * (y * z + w * x)).clamp(-1, 1))).cpu()
  for e in mv.nonzero().flatten().tolist():
    rolls[e].append(abs(float(r[e])))
  fpos = robot.data.body_link_pos_w[:, st.foot_ids, :2]
  dd = fpos[:, 0] - fpos[:, 1]
  wl = (-dd[:, 0] * hf[:, 1] + dd[:, 1] * hf[:, 0]).abs().cpu()
  wmin = torch.where(mv, torch.minimum(wmin, wl), wmin)
  stopped = (t >= MOVE) & alive
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

slip += ((prev_f - anchor).norm(dim=-1) * prev_lo).sum(-1)
vm = vsum / vcnt.clamp(min=1)[:, None]
print(f"OSEQ {label}:")
for i, (name, c) in enumerate(PAT):
  ids = [e for e in range(N) if e % len(PAT) == i]
  ok = [e for e in ids if not fell[e]]
  cm = torch.tensor(c) != 0
  if ok:
    ratio = [float((vm[ok][:, j] / tgt[ok][:, j]).mean()) for j in range(3) if cm[j]]
    leak = [float(vm[ok][:, j].abs().mean()) for j in range(3) if not cm[j]]
    rp = float(np.percentile(np.concatenate([np.array(rolls[e]) for e in ok if rolls[e]] or [np.zeros(1)]), 95))
    print(f"  {name:13s} chute {int(fell[ids].sum())}/{len(ids)} | ratio " + " ".join(f"{x:.2f}" for x in ratio)
          + " | fuite " + " ".join(f"{x:.3f}" for x in leak) + f" | roulis p95 {rp:.1f} | largeur min {float(wmin[ok].min()) * 100:.0f} cm"
          + f" | arret glisse med {float(slip[ok].median()) * 100:.1f} cm poses {float(lands[ok].median()):.0f}")
  else:
    print(f"  {name:13s} chute {int(fell[ids].sum())}/{len(ids)}")
