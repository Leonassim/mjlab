"""s50smoke.py: build the training env (CPU, 64 envs), check 50 Hz wiring, step random actions, report per-term totals."""
import os, math
os.environ.setdefault("RHPS1_STAIRS", "1")
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.config.rhps1 import stairs as ST
cfg = load_env_cfg("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start")
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
cfg.scene.num_envs = int(os.environ.get("N", "64"))
torch.manual_seed(0)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=12.0)
u = env.unwrapped; st = u._stairs
print("step_dt", u.step_dt, "decimation", u.cfg.decimation, "physics dt", u.cfg.sim.mujoco.timestep)
print("vz_buf", tuple(st.vz_buf.shape), "feet history", getattr(u.scene["stairs_feet"].cfg, "history_length", None))
print("envs: RL", int(st.is_stairs.sum()), "pyramid", int(st.pyr.sum()), "flat RL", int((st.is_stairs & ~st.pyr).sum()), "standing", int(st.standing.sum()))
print("max episode steps (stairs)", math.ceil(ST.EPISODE_S / u.step_dt), "| rewards", len(u.reward_manager.active_terms))
obs = env.get_observations()
names = u.reward_manager.active_terms
tot = {k: 0.0 for k in names}; nan = 0; peaks = []; resets = 0
T = int(6.0 / u.step_dt)
for i in range(T):
  a = torch.zeros(u.num_envs, u.action_manager.total_action_dim) if os.environ.get('ZERO') else 0.3 * torch.randn(u.num_envs, u.action_manager.total_action_dim)
  obs, rew, dones, extras = env.step(a)
  nan += int(torch.isnan(rew).sum())
  resets += int(dones.sum())
  for k in names:
    tot[k] += float(u.reward_manager._step_reward[:, names.index(k)].sum()) if hasattr(u.reward_manager, "_step_reward") else 0.0
  peaks.append(float(st.f_peak.max()))
print("steps", T, "nan rewards", nan, "resets", resets, "f_peak max", max(peaks), "obs actor", tuple(obs["actor"].shape))
print("flat RL envs: goal-s mean", float(((((u.scene['robot'].data.root_link_pos_w[:, :2] - st.center) * st.u).sum(-1) - st.goal) * st.sign)[st.is_stairs & ~st.pyr].mean()) if (st.is_stairs & ~st.pyr).any() else None)
print("progress finite", bool(torch.isfinite(st.progress).all()), "standing now", int(st.standing.sum()), "cmd_on huge", int((st.cmd_on > 10**8).sum()))
rate = {k: v / (T * u.num_envs) for k, v in tot.items()}
print("RATES " + " ".join(f"{k}={rate[k]:+.4f}" for k in names))
