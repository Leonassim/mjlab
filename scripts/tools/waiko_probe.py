"""MCWaiko emulation (RHPS1_WAIKO=1) vs the truth: per-direction error of the actor's
base_lin_vel source, to compare with the real MCWaiko in mc_mujoco (0.09 m/s rms
walking, ~40 ms late, ~15% short).

Adapted from robust_probe.py (falls under perturbation):

  RHPS1_WAIKO=1 uv run python scripts/tools/waiko_probe.py <run> <ckpt>

Each env holds one command from rest (standing, forward, lateral left, yaw left,
backward); a fall is the pelvis below 0.55 m. The command reaches the policy
through the controller's filter (RHPS1_CMD_TAU), as on the robot.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
CMDS = {
  "debout": (0, 0, 0),
  "avant": (1, 0, 0),
  "gauche": (0, 1, 0),
  "rot+": (0, 0, 1),
  "arriere": (-1, 0, 0),
}
PER = int(os.environ.get("PROBE_PER", "4"))
N = PER * len(CMDS)
torch.manual_seed(int(os.environ.get("PROBE_SEED", "0")))
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = N
cfg.terminations = {}
cfg.episode_length_s = 1e9
cfg.commands["motion"].motion_file = os.environ.get(
  "MJLAB_MOTION_FILE", "docs/bwc_motion_v3.npz"
)
# No reference resets mid-run: every env starts at frame 0 of the first clip
# (long enough for 23 s), as in play. Otherwise RSI and the clip seam teleport
# envs and fake both falls and drift.
_m = cfg.commands["motion"]
_m.sampling_mode = "start"
_m.pose_range = {}
_m.velocity_range = {}
if hasattr(_m, "standing_start_prob"):
  _m.standing_start_prob = 0.0
_lag = int(
  round(float(os.environ.get("FLIP_DELAY_MS", "30")) / 1000 / cfg.sim.mujoco.timestep)
)
for _a in cfg.scene.entities["robot"].articulation.actuators:
  _a.delay_min_lag = _lag
  _a.delay_max_lag = _lag
TARGET = torch.tensor(
  [c for c in CMDS.values() for _ in range(PER)], dtype=torch.float32
)
CMD = torch.zeros(N, 3)
TAU = float(os.environ.get("RHPS1_CMD_TAU", "0.3"))
_filt = {"y": torch.zeros(N, 3), "step": -1}


def _cmd(env, command_name, tau=0.0):
  step = int(env.common_step_counter)
  if _filt["step"] != step:
    _filt["y"] = _filt["y"] + (env.step_dt / TAU) * (CMD - _filt["y"])
    _filt["step"] = step
  return _filt["y"].to(env.device)


cfg.observations["actor"].terms["velocity_command"].func = _cmd
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(
  str(Path("logs/rsl_rl/rhps1_tracking") / run / ck),
  load_cfg={"actor": True},
  strict=True,
  map_location="cpu",
)
pol = runner.get_inference_policy(device="cpu")
STOCH = os.environ.get("PROBE_STOCHASTIC") == "1"  # sample actions as in training
CMD_T = float(os.environ.get("PROBE_CMD_T", "3.0"))  # command step time; arming at 1 s
HOLD = torch.zeros(N, env.num_actions)
if os.environ.get("PROBE_HOLD_POSTURE"):
  # mc_mujoco's arming: hold the stance posture, then seed that action as the last one.
  import json

  _post = json.load(open(os.environ["PROBE_HOLD_POSTURE"]))
  _term = u.action_manager.get_term("joint_pos")
  _rb = u.scene["robot"]
  _names = (
    list(_term._joint_names)
    if hasattr(_term, "_joint_names")
    else list(_rb.joint_names)
  )
  _q0 = _rb.data.default_joint_pos[0]
  _jn = list(_rb.joint_names)
  _sc = (
    _term._scale[0]
    if torch.is_tensor(_term._scale)
    else torch.full((env.num_actions,), _term._scale)
  )
  for _a, _n in enumerate(_names):
    if _n in _post:
      HOLD[:, _a] = (_post[_n] - float(_q0[_jn.index(_n)])) / float(_sc[_a])
  print(
    "hold action max",
    float(HOLD.abs().max()),
    "on",
    _names[int(HOLD[0].abs().argmax())],
  )
robot = u.scene["robot"]
dt = u.step_dt
obs, _ = env.reset()
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(
  q0 + (torch.rand_like(q0) * 2 - 1) * 0.03, torch.zeros_like(q0)
)
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)
fell = torch.zeros(N, dtype=torch.bool)
arm_peak = torch.zeros(
  N
)  # joint speed in the second after arming (controller guard: 4 rad/s)
start = robot.data.root_link_pos_w[:, :2].clone()


def _yaw():
  q = robot.data.root_link_quat_w
  return torch.atan2(
    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
  )


EST, TRU = [], []
with torch.no_grad():
  for k in range(int(20.0 / dt)):
    CMD[:] = TARGET if k * dt >= CMD_T else 0.0
    obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else HOLD)
    est = u._waiko_v.clone()
    v = robot.data.root_link_lin_vel_w
    y = _yaw()
    c, s = torch.cos(y), torch.sin(y)
    TRU.append(torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]), -1).numpy())
    EST.append(est[:, :2].numpy())
EST, TRU = np.array(EST), np.array(TRU)  # (T, N, 2)
k0 = int((CMD_T + 1.5) / dt)
print(f"{run} {ck}  emulation MCWaiko tau {os.environ.get('RHPS1_WAIKO_TAU', '0.08')} s scale {os.environ.get('RHPS1_WAIKO_SCALE', '0.82,0.98')}")
for i, name in enumerate(CMDS):
  sl = slice(i * PER, (i + 1) * PER)
  e, t = EST[k0:, sl], TRU[k0:, sl]
  err = e - t
  rms = np.sqrt((err ** 2).sum(-1).mean())
  trms = np.sqrt((t ** 2).sum(-1).mean())
  dist_t = np.linalg.norm(t.sum(0) * dt, axis=-1).mean()
  dist_e = np.linalg.norm(e.sum(0) * dt, axis=-1).mean()
  a = (e - e.mean(0)).reshape(len(e), -1); b = (t - t.mean(0)).reshape(len(t), -1)
  lags = list(range(-30, 31))
  cc = [np.sum(a[max(0, -L):len(a) - max(0, L)] * b[max(0, L):len(b) - max(0, -L)]) for L in lags]
  lag = -lags[int(np.argmax(cc))] * dt * 1000
  print(f"  {name:8s} erreur rms {rms:.3f} m/s (vitesse vraie rms {trms:.3f}) | distance estimee {dist_e:.2f} / vraie {dist_t:.2f} m"
        f" ({100 * (dist_e / max(dist_t, 1e-6) - 1):+.0f}%) | retard {lag:+.0f} ms")
