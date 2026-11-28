"""stairs_budget.py <run|ckpt dir> <ckpt> [label]: per-second reward budget on stairs vs flat tiles at one curriculum level.

LEVEL (default 2) is forced on every env, FRAC (default 0.5) splits stairs/flat tiles, pauses off, the task's own
commands. Each term's rate (weight x value per second) is averaged per env over T s from 1 s on, then over three
groups: stairs envs that climb (mean forward speed > 2 cm/s), stairs envs that stall, flat envs. A term that bills
the climbers much more than the stallers is what makes standing in front of the riser pay. Recipe env, CPU.
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

LV = os.environ.get("LEVEL", "2")
os.environ.update({"RHPS1_STAIRS_INIT_LEVEL": LV, "RHPS1_STAIRS_LVL_MAX": LV, "RHPS1_STAIRS_LVL_FILE": "",
                   "RHPS1_STAIRS_FRAC": os.environ.get("FRAC", "0.5"), "RHPS1_STAIRS_PAUSE_P": "0",
                   "RHPS1_STAIRS_STAIR_PAUSE_P": "0", "RHPS1_STAIRS_STAND_P": "0",
                   "RHPS1_STAIRS_APPROACH_RAMP": "1000000000"})
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
N, T = int(os.environ.get("BENCH_N", "64")), float(os.environ.get("T", "16"))
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=12.0)
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path(sys.argv[1]) / sys.argv[2]), load_cfg={"actor": True}, strict=True, map_location="cpu")
pol = runner.alg.actor
u.reset()
obs = env.get_observations()
rm, dt, robot = u.reward_manager, u.step_dt, u.scene["robot"]
names = rm.active_terms
acc = torch.zeros(N, len(names))
vx = torch.zeros(N)
x0 = robot.data.root_link_pos_w[:, :2].cpu().clone()
dmax = torch.zeros(N)
fid = u._stairs.foot_ids
zmax = torch.zeros(N)
alive = torch.ones(N, dtype=torch.bool)
n = 0
for k in range(int(T / dt)):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  alive &= ~u.termination_manager.terminated.cpu()
  if k * dt >= 1.0:
    acc += rm.term_rewards.cpu() / dt
    vx += robot.data.root_link_lin_vel_b[:, 0].cpu()
    dmax = torch.maximum(dmax, (robot.data.root_link_pos_w[:, :2].cpu() - x0).norm(dim=-1))
    fz = robot.data.body_link_pos_w[:, fid, 2].cpu()
    zmax = torch.maximum(zmax, (fz.max(1).values - fz.min(1).values))
    n += 1
acc, vx = acc / n, vx / n
pyr = u._stairs.pyr.cpu().bool()
groups = {"stairs climb": pyr & alive & (vx > 0.02), "stairs stall": pyr & alive & (vx <= 0.02),
          "flat": ~pyr & alive}
lab = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
print(f"{lab} level {LV}: " + ", ".join(f"{g} {int(m.sum())}" for g, m in groups.items()) + f", falls {int((~alive).sum())}/{N}")
M = {g: acc[m].mean(0) if m.any() else torch.full((len(names),), float("nan")) for g, m in groups.items()}
order = np.argsort(-np.abs((M["stairs climb"] - M["stairs stall"]).nan_to_num().numpy()))
print(f"  {'term':28s} " + " ".join(f"{g:>13s}" for g in groups) + "   climb-stall  (per s)")
for i in order[:18]:
  print(f"  {names[i]:28s} " + " ".join(f"{M[g][i]:+13.2f}" for g in groups) + f"   {M['stairs climb'][i] - M['stairs stall'][i]:+9.2f}")
print("  " + " | ".join(f"{g}: travel {100 * dmax[m].mean():.0f} cm, max foot-to-foot height {100 * zmax[m].mean():.1f} cm" for g, m in groups.items() if m.any()))
print("  total".ljust(31) + " ".join(f"{M[g].sum():+13.2f}" for g in groups))
