"""budget.py <run|ckpt dir> <ckpt> [label]: per-second reward budget of a flat walker under one held command.

Every env holds CMD ("vx,vy,wz" signs, default forward) for T s; from 1 s on, each reward term's rate (weight x value,
per second, as Episode_Reward but per state) is averaged over all steps, and split by double support (both soles >
10 % BW) vs single support. Shows what a gait change costs: terms billed per stance-second favour short stances.
ACT=sto adds the policy's noise. Run with the recipe's env (diag_run-style) on CPU.
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

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else f"{run}/{ck}"
N = int(os.environ.get("BENCH_N", "16"))
T = float(os.environ.get("DIAG_T", "8"))
CMD = [float(x) for x in os.environ.get("CMD", "1,0,0").split(",")]
ACT = os.environ.get("ACT", "det")
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
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
p = Path(run)
if not p.is_absolute():
  p = Path("checkpoints/rhps1") / run if (Path("checkpoints/rhps1") / run).is_dir() else Path("logs/rsl_rl/rhps1_tracking") / run
runner.load(str(p / ck), load_cfg={"actor": True}, strict=True, map_location=dev)
pol = runner.alg.actor
u.reset()
obs = env.get_observations()
st.oseq[:] = True
st.seg_cmd[:] = 0.0
st.seg_cmd[:, 0] = torch.tensor(CMD, device=dev)
st.seg_end[:] = 10**9
rm = u.reward_manager
names = list(rm.active_terms)
BW = 57.64 * 9.81
dt = u.step_dt
acc = {k: [] for k in ("all", "ds", "ss")}
alive = torch.ones(N, dtype=torch.bool)
for k in range(int(T / dt)):
  with torch.no_grad():
    a = pol(obs, stochastic_output=True) if ACT == "sto" else pol(obs)
    obs, _, _, _ = env.step(a)
  alive &= ~u.termination_manager.terminated.cpu()
  if k * dt < 1.0:
    continue
  r = rm._step_reward.cpu().numpy()[alive.numpy()]
  ld = (st.load.cpu() > 0.1 * BW).numpy()[alive.numpy()]
  ds = ld.all(-1)
  acc["all"].append(r)
  acc["ds"].append(r[ds])
  acc["ss"].append(r[~ds])
A = {k: np.concatenate(v) for k, v in acc.items()}
share_ds = len(A["ds"]) / max(len(A["all"]), 1)
print(f"BUDGET {label} CMD={CMD} ACT={ACT} | alive {int(alive.sum())}/{N} | double support {100 * share_ds:.0f} % of steps")
print(f"  {'term':28s} {'all /s':>8s} {'in DS /s':>9s} {'in SS /s':>9s}")
order = np.argsort(-np.abs(A["all"].mean(0)))
tot = {k: A[k].mean(0) if len(A[k]) else np.zeros(len(names)) for k in A}
for i in order:
  if abs(tot["all"][i]) < 0.02:
    continue
  print(f"  {names[i]:28s} {tot['all'][i]:+8.2f} {tot['ds'][i]:+9.2f} {tot['ss'][i]:+9.2f}")
print(f"  {'TOTAL':28s} {tot['all'].sum():+8.2f} {tot['ds'].sum():+9.2f} {tot['ss'].sum():+9.2f}")
