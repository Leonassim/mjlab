"""omni_eval.py <run> <ckpt> [label]: side-step and turn-in-place bench on flat ground (RHPS1_STAIRS_OMNI_P).

Every env is a flat tile with an omni episode (vy = +-1 or yaw = +-1 until the goal, then stop), deterministic
policy, no pauses. Per mode: arrivals, falls, mean speed while commanded (lateral m/s or yaw rad/s, in the commanded
direction), time to arrive, drift after arrival (pelvis speed and yaw rate over the last 2 s), forward drift while
side-stepping or turning.
  BENCH_N=64 BENCH_T=16 python omni_eval.py <run> <ckpt> [label]
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
os.environ["RHPS1_STAIRS_OMNI_P"] = os.environ.get("RHPS1_STAIRS_OMNI_P", "1")
for k in ("RHPS1_STAIRS_FRAC", "RHPS1_STAIRS_STAND_P", "RHPS1_STAIRS_PAUSE_P", "RHPS1_STAIRS_STAIR_PAUSE_P"):
  os.environ[k] = "0"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ck
N = int(os.environ.get("BENCH_N", "64"))
T = float(os.environ.get("BENCH_T", "16"))
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
mode = st.omni.clone().cpu()
odir = st.odir.clone().cpu()

alive = torch.ones(N, dtype=torch.bool)
fell = torch.zeros(N, dtype=torch.bool)
t_arr = torch.full((N,), float("nan"))
v_on = [[] for _ in range(N)]
fwd_on = [[] for _ in range(N)]
drift = [[] for _ in range(N)]
steps = int(T / u.step_dt)
for i in range(steps):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  term = u.termination_manager.terminated.cpu()
  fell |= term & alive
  alive &= ~term
  arr = st.arrived.cpu()
  t_arr = torch.where(arr & torch.isnan(t_arr) & alive, torch.full_like(t_arr, i * u.step_dt), t_arr)
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  hf = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
  v = robot.data.root_link_lin_vel_w[:, :2]
  wz = robot.data.root_link_ang_vel_w[:, 2].cpu()
  fwd = (v * hf).sum(-1).cpu()
  lat = (v[:, 0] * -hf[:, 1] + v[:, 1] * hf[:, 0]).cpu()
  on = (u.episode_length_buf >= st.cmd_on).cpu() & ~arr & alive
  late = arr & alive & (i * u.step_dt >= T - 2.0)
  for e in range(N):
    if on[e]:
      v_on[e].append(float((lat[e] if mode[e] == 1 else wz[e]) * odir[e]))
      fwd_on[e].append(float(fwd[e]))
    if late[e]:
      drift[e].append((float(v[e].norm()), abs(float(wz[e]))))

print(f"OMNI {label}:")
for m, name, unit in ((1, "cote", "m/s"), (2, "tourne", "rad/s")):
  ids = (mode == m).nonzero().flatten().tolist()
  if not ids:
    continue
  arr = [e for e in ids if not torch.isnan(t_arr[e]) and not fell[e]]
  vs = [x for e in ids for x in v_on[e]]
  fw = [abs(x) for e in ids for x in fwd_on[e]]
  dr = [d for e in ids for d in drift[e]]
  ta = [float(t_arr[e]) for e in arr]
  print(f"  {name}: arr {len(arr)}/{len(ids)} chute {int(fell[ids].sum())}"
        f" | vitesse commandee moy {sum(vs) / max(len(vs), 1):+.3f} {unit}"
        f" | avance parasite |vx| moy {sum(fw) / max(len(fw), 1):.3f} m/s"
        f" | temps d'arrivee moy {sum(ta) / max(len(ta), 1):.1f} s"
        f" | apres arret (2 s) v moy {sum(d[0] for d in dr) / max(len(dr), 1):.3f} m/s"
        f" lacet moy {sum(d[1] for d in dr) / max(len(dr), 1):.3f} rad/s")
