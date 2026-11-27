"""rest_cop_probe.py <ckpt dir> <ckpt> [label]: walk forward WALK_S, then zero command; over the last 3 s of REST_S
print each sole's lateral CoP in the ankle frame (cm, + = robot's left; ~6.4 cm outward is the frame's geometric
offset), the ankle width, the body sole roll and hip/ankle roll vs the real stance. Recipe env, CPU."""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("RHPS1_STAIRS", "1")
os.environ["RHPS1_STAIRS_OSEQ_P"] = "1"
for k in ("RHPS1_STAIRS_FRAC", "RHPS1_STAIRS_STAND_P"):
  os.environ[k] = "0"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
N = int(os.environ.get("BENCH_N", "16"))
WALK, REST = float(os.environ.get("WALK_S", "4")), float(os.environ.get("REST_S", "6"))
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = os.environ.get("MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz")
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=12.0)
u = env.unwrapped
st = u._stairs
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path(sys.argv[1]) / sys.argv[2]), load_cfg={"actor": True}, strict=True, map_location="cpu")
pol = runner.alg.actor
u.reset()
obs = env.get_observations()
st.oseq[:] = True
st.seg_cmd[:] = 0.0
st.seg_cmd[:, 0] = torch.tensor([1.0, 0.0, 0.0])
st.seg_end[:] = 10**9
dt = u.step_dt
ys, rolls, still, qs, widths, tps, tfs = [], [], [], [], [], [], []
am = u.action_manager.get_term(u.action_manager.active_terms[0])
robot = u.scene["robot"]
jn = robot.joint_names
alive = torch.ones(N, dtype=torch.bool)
for k in range(int((WALK + REST) / dt)):
  if k == int(WALK / dt):
    st.seg_cmd[:, 0] = 0.0
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  alive &= ~u.termination_manager.terminated.cpu()
  if k * dt > WALK + REST - 3:
    pos, _, _, _, fn = ST._feet(u)
    fn = fn.view(N, 2, -1)
    cop = (pos.view(N, 2, -1, 3) * fn[..., None]).sum(-2) / fn.sum(-1).clamp(min=1e-6)[..., None]
    qf = robot.data.body_link_quat_w[:, st.foot_ids]
    w, x, y_, z_ = qf.unbind(-1)
    fy = torch.stack((2 * (x * y_ - w * z_), 1 - 2 * (x * x + z_ * z_), 2 * (y_ * z_ + w * x)), -1)
    ap = robot.data.body_link_pos_w[:, st.foot_ids]
    ys.append(((cop - ap) * fy).sum(-1)[alive].numpy())
    yaw = torch.atan2(2 * (robot.data.root_link_quat_w[:, 0] * robot.data.root_link_quat_w[:, 3]), 1 - 2 * robot.data.root_link_quat_w[:, 3] ** 2)
    d = ap[:, 0, :2] - ap[:, 1, :2]
    widths.append((-d[:, 0] * torch.sin(yaw) + d[:, 1] * torch.cos(yaw)).abs()[alive].numpy())
    rolls.append(torch.rad2deg(ST._foot_tilt(u)[1])[alive].numpy())
    still.append(ST._still(u)[alive].float().mean().item())
    qs.append(robot.data.joint_pos[alive].numpy())
    tf = robot.data.joint_pos_target.clone(); tfs.append(tf[alive].numpy())
    tp = tf.clone(); tp[:, am._target_ids] = am._processed_actions; tps.append(tp[alive].numpy())
y = 100 * np.concatenate(ys)
r = np.concatenate(rolls)
print(f"{sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]}: alive {int(alive.sum())}/{N}, still {100 * np.mean(still):.0f} % | rest lateral CoP L {y[:, 0].mean():+.2f} R {y[:, 1].mean():+.2f} cm"
      f" (geometric offset ~6.4 outward) | ankle width {100 * np.concatenate(widths).mean():.1f} cm | body sole roll L {r[:, 0].mean():+.2f} R {r[:, 1].mean():+.2f} deg")
import math
Q = np.degrees(np.concatenate(qs).mean(0))
out = []
for n in ("L_CROTCH_R", "R_CROTCH_R", "L_ANKLE_R", "R_ANKLE_R"):
  k = n[2:]
  ref = (-1 if n.startswith("R_") else 1) * ST.STANCE_DEG[k]
  out.append(f"{n} {Q[jn.index(n)] - ref:+.2f}")
print("  rest q - real stance (deg): " + " ".join(out))
LEGS = [f"{s}_{j}" for s in "LR" for j in ("CROTCH_Y", "CROTCH_R", "CROTCH_P", "KNEE_P", "ANKLE_R", "ANKLE_P")]
TP, TF = (np.degrees(np.concatenate(a).mean(0)) for a in (tps, tfs))
print("  rest deg (policy target / actuator target / measured): " + " | ".join(f"{n} {TP[jn.index(n)]:+.2f}/{TF[jn.index(n)]:+.2f}/{Q[jn.index(n)]:+.2f}" for n in LEGS))
