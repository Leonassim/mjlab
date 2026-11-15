"""apsmoke.py <run> <ckpt>: spawn distance/yaw and heading command check in the training env, deterministic policy."""
import os, sys, math
from dataclasses import asdict
from pathlib import Path
import torch
os.environ.setdefault("RHPS1_STAIRS", "1")
from mjlab.envs import ManagerBasedRlEnv  # noqa
from mjlab.rl import RslRlVecEnvWrapper  # noqa
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa
TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
cfg = load_env_cfg(TASK); cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"; cfg.scene.num_envs = int(os.environ.get("SMOKE_N", "16"))
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=12.0)
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / sys.argv[1] / sys.argv[2]), load_cfg={"actor": True}, strict=True, map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
u = env.unwrapped; st = u._stairs; robot = u.scene["robot"]
obs = env.get_observations()
print("episode steps", int(u.max_episode_length), "GAP", ST.GAP, "YAW_SPAWN deg", round(math.degrees(ST.YAW_SPAWN), 1), "HEAD_CMD", ST.HEAD_CMD, "ARRIVE_DESC", ST.ARRIVE_DESC)
span = ST.PLATFORM / 2 + (st.nsteps - 1) * st.tread
def yaw_err():
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  d = torch.atan2(st.face[:, 1], st.face[:, 0]) - yaw
  return torch.rad2deg(torch.atan2(torch.sin(d), torch.cos(d)))
print("spawn check:", " | ".join(f"{'C' if st.climb[e] else 'D'} s0 {float(st.s0[e]):.2f} trig {float(st.sp_trig[e]):.2f} span {float(span[e]):.2f}" for e in range(min(6, u.num_envs))))
vc = [k for k in u.observation_manager.active_terms["actor"]].index("velocity_command")
for i in range(int(float(os.environ.get("SMOKE_T", "12")) / u.step_dt)):
  if i % int(float(os.environ.get("SMOKE_DT", "1")) / u.step_dt) == 0:
    sp = ((robot.data.root_link_pos_w[:, :2] - st.center) * st.u).sum(-1) - span
    cmd = ST.E._reference_velocity_command(u, "motion") if hasattr(ST, "E") else None
    from mjlab.tasks.tracking.config.rhps1 import env_cfgs as E
    c = E._reference_velocity_command(u, "motion")
    print(f"t {i * u.step_dt:4.1f}", " | ".join(f"{'C' if st.climb[e] else 'D'} s {float(sp[e]):+.2f} yaw {float(yaw_err()[e]):+4.0f} cmd {int(c[e,0])}{int(c[e,2]):+d} lvl {int(u.scene.terrain.terrain_levels[e])}{' P' if st.paused[e] else ''}{' A' if st.arrived[e] else ''}" for e in range(min(6, u.num_envs))))
  with torch.no_grad():
    obs, *_ = env.step(pol(obs))
  if i == 0:
    ARR = torch.zeros(u.num_envs, device="cpu"); FALL = torch.zeros(u.num_envs, device="cpu"); DIST = torch.zeros(u.num_envs, device="cpu")
    prev_s = ((robot.data.root_link_pos_w[:, :2] - st.center) * st.u).sum(-1)
  ARR += st.arrived_now.float().cpu()
  FALL += u.termination_manager.get_term("stairs_fall").float().cpu() if "stairs_fall" in u.termination_manager.active_terms else 0
  s_now = ((robot.data.root_link_pos_w[:, :2] - st.center) * st.u).sum(-1)
  fresh = (u.episode_length_buf <= 1)
  DIST += torch.where(fresh, torch.zeros_like(s_now), (st.sign * (s_now - prev_s))).clamp(-0.05, 0.05).cpu()
  prev_s = s_now

T_tot = float(os.environ.get("SMOKE_T", "12"))
print(f"SUMMARY {sys.argv[2]}: arrivals {int(ARR.sum())} falls {int(FALL.sum())} over {u.num_envs} envs x {T_tot:.0f} s | forward speed mean {float(DIST.mean()) / T_tot:.3f} m/s, envs under 1 cm/s {int((DIST / T_tot < 0.01).sum())}")
