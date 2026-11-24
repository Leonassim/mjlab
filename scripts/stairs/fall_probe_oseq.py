"""Forward command from standing: pelvis pitch/roll, height, forward speed, foot loads, until the fall."""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

os.environ["RHPS1_STAIRS_OSEQ_P"] = "1"
for k in ("RHPS1_STAIRS_FRAC", "RHPS1_STAIRS_STAND_P", "RHPS1_STAIRS_PAUSE_P", "RHPS1_STAIRS_STAIR_PAUSE_P"):
  os.environ[k] = "0"
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
cmd = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,0,0").split(",")]
N, dev = 16, "cuda:0"
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
cfg.episode_length_s = 1e9
cfg.curriculum = {}
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "stairs_fall")}
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=dev), clip_actions=12.0)
u = env.unwrapped
st = u._stairs
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=dev)
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True}, strict=True,
            map_location=dev)
pol = runner.get_inference_policy(device=dev)
u.reset()
obs = env.get_observations()
st.oseq[:] = True
st.seg_cmd[:] = 0.0
st.seg_cmd[:, 0] = torch.tensor(cmd, device=dev)
st.seg_end[:] = 10**9
robot = u.scene["robot"]
dt = u.step_dt
alive = torch.ones(N, dtype=torch.bool, device=dev)
tfall = torch.full((N,), float("nan"))
print(f"cmd {cmd}, cmd_on med {float(st.cmd_on.float().median()) * dt:.2f} s")
print(" t(s)  alive  pitch med  roll|med  z med   vfwd med  load L/R med (BW)  lift max cm")
BW = 57.64 * 9.81
for i in range(int(8 / dt)):
  with torch.no_grad():
    obs, _, _, _ = env.step(pol(obs))
  term = u.termination_manager.terminated
  newf = term & alive
  tfall[newf.cpu()] = i * dt
  alive &= ~term
  if i % int(0.5 / dt) == 0 and alive.any():
    q = robot.data.root_link_quat_w
    w, x, y, z = q.unbind(-1)
    pitch = torch.rad2deg(torch.asin((2 * (w * y - z * x)).clamp(-1, 1)))
    roll = torch.rad2deg(torch.asin((2 * (y * z + w * x)).clamp(-1, 1)))
    a = alive
    zz = robot.data.root_link_pos_w[:, 2]
    v = robot.data.root_link_lin_vel_w[:, :2]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    vf = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)
    ld = st.load / BW
    hs = st.hmax * 100
    print(f"{i * dt:5.1f}  {int(a.sum()):5d}  {float(pitch[a].median()):8.1f}  {float(roll[a].abs().median()):7.1f}"
          f"  {float(zz[a].median()):5.3f}  {float(vf[a].median()):8.3f}  {float(ld[a, 0].median()):.2f}/"
          f"{float(ld[a, 1].median()):.2f}  {float(hs[a].max()):6.1f}")
print("fall times (s):", sorted(round(float(x), 1) for x in tfall if x == x))
