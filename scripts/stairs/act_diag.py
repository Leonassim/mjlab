"""act_diag.py <run|-> <ckpt|-> [label]: what the exploration noise does to the legs (position-control diagnosis).

Flat, forward command, BENCH_N envs for DIAG_T s. ACT = det (policy mean), sto (mean + the policy's own noise), zero,
noise (zero mean + DIAG_STD white noise; no checkpoint needed). Per leg joint group:
  dem   PD demand |kp (q* - q) + kd (qd* - qd)| / effort p50 / p90, from the actuator's applied force
  sat   share of steps at >= 95 % of the effort limit
  jit   joint jitter: rms of q minus its DIAG_W (0.1 s) moving average (deg), what the noise shakes
  slow  rms of that moving average minus its episode mean (deg), the motion the legs actually make
and per robot: foot lift (p90 of sole height over its own floor, cm), single support share, falls.
  probe.sh act_diag.py gi_1000 model_1000.pt -- ACT=sto        (checkpoints/rhps1/<name> or a run dir)
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
for k in ("RHPS1_STAIRS_FRAC", "RHPS1_STAIRS_STAND_P", "RHPS1_STAIRS_PAUSE_P", "RHPS1_STAIRS_STAIR_PAUSE_P",
          "RHPS1_STAIRS_OMNI_P", "RHPS1_STAIRS_OSEQ_P"):
  os.environ[k] = "0"
os.environ["RHPS1_STAIRS_GAP"] = "2.5,2.5"
os.environ["RHPS1_STAIRS_YAW_SPAWN"] = "0"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else f"{run}/{ck}"
ACT = os.environ.get("ACT", "det")
N = int(os.environ.get("BENCH_N", "16"))
T = float(os.environ.get("DIAG_T", "8"))
STD = float(os.environ.get("DIAG_STD", "1.0"))
dev = os.environ.get("BENCH_DEV", "cpu")
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
pol = None
if ck != "-":
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
  p = Path(run)
  if not p.is_absolute():
    p = Path("checkpoints/rhps1") / run if (Path("checkpoints/rhps1") / run).is_dir() else Path("logs/rsl_rl/rhps1_tracking") / run
  runner.load(str(p / ck), load_cfg={"actor": True}, strict=True, map_location=dev)
  pol = runner.alg.actor
u.reset()
obs = env.get_observations()
robot = u.scene["robot"]
jn = list(robot.joint_names)
GROUPS = ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE_P", "ANKLE_P", "ANKLE_R")
EFF = {"CROTCH_Y": 35.0, "CROTCH_R": 100.0, "CROTCH_P": 140.0, "KNEE_P": 70.0, "ANKLE_P": 65.0, "ANKLE_R": 45.0}
gid = {g: [i for i, n in enumerate(jn) if n.endswith(g)] for g in GROUPS}
fids = st.foot_ids
BW = 57.64 * 9.81
dt = u.step_dt
steps = int(T / dt)
Q, F, Z, L = [], [], [], []
alive = torch.ones(N, dtype=torch.bool)
fell = torch.zeros(N, dtype=torch.bool)
std_seen = None
for k in range(steps):
  with torch.no_grad():
    if pol is None:
      a = torch.zeros(N, u.action_manager.total_action_dim, device=dev)
      if ACT == "noise":
        a = a + STD * torch.randn_like(a)
    elif ACT == "sto":
      a = pol(obs, stochastic_output=True)
      std_seen = float(pol.output_std.mean())
    elif ACT == "zero":
      a = torch.zeros(N, u.action_manager.total_action_dim, device=dev)
    else:
      a = pol(obs)
    obs, _, _, _ = env.step(a)
  term = u.termination_manager.terminated.cpu()
  fell |= term & alive
  alive &= ~term
  Q.append(robot.data.joint_pos.cpu().clone())
  F.append(robot.data.qfrc_actuator.cpu().clone())
  Z.append(robot.data.body_link_pos_w[:, fids, 2].cpu().clone())
  L.append(st.load.cpu().clone())
Q, F, Z, L = (torch.stack(x).numpy() for x in (Q, F, Z, L))  # [T, N, ...]
ok = (~fell).numpy()
w = max(1, int(float(os.environ.get("DIAG_W", "0.1")) / dt))
t0 = int(1.0 / dt)  # skip the first second (hold, start)


def movavg(x):
  c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], 0), 0)
  m = (c[w:] - c[:-w]) / w
  return np.concatenate([np.repeat(m[:1], w - 1, 0), m], 0)


print(f"ACTDIAG {label} ACT={ACT}" + (f" std={STD}" if ACT == "noise" else "") + (f" policy std {std_seen:.3f}" if std_seen else "")
      + f" | falls {int(fell.sum())}/{N}")
print("  group      dem p50  p90   sat %   jit deg  slow deg")
for g in GROUPS:
  ids = gid[g]
  if not ids or not ok.any():
    continue
  q = Q[t0:, :, ids][:, ok]
  f = np.abs(F[t0:, :, ids][:, ok]) / EFF[g]
  ma = movavg(Q[:, :, ids])[t0:, ok]
  jit = np.degrees(np.sqrt(np.mean((q - ma) ** 2)))
  slow = np.degrees(np.sqrt(np.mean((ma - ma.mean(0, keepdims=True)) ** 2)))
  print(f"  {g:9s} {np.percentile(f, 50):7.2f} {np.percentile(f, 90):5.2f} {100 * np.mean(f >= 0.95):6.1f} {jit:9.2f} {slow:9.2f}")
if ok.any():
  z = Z[t0:, ok]
  lift = z - np.percentile(z, 5, axis=0, keepdims=True)
  ld = L[t0:, ok] > 0.1 * BW
  ss = np.mean(ld.sum(-1) == 1)
  print(f"  foot lift p90 {100 * np.percentile(lift, 90):.1f} cm, max {100 * np.max(lift):.1f} cm | single support {100 * ss:.0f} %")
  from mjlab.tasks.tracking.config.rhps1.hold_action import SLOT18_P99_VEL

  sc = np.array([SLOT18_P99_VEL.get(n, 1.0) for n in jn])
  v = np.diff(Q[t0:, ok], axis=0) / dt
  vr = (((np.diff(v, axis=0)) / sc) ** 2).sum(-1)
  print(f"  stairs_vrate (sum of (dqd/slot18 p99)^2 per step): mean {vr.mean():.2f} p95 {np.percentile(vr, 95):.2f} (slot 18: 0.64)")
