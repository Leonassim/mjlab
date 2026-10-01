"""Command onset while standing in control: the operator's joystick push.

Settle 3 s with actions held, then the policy stands at command 0 for 2-5 s
(drawn per env) before the command flips to forward. Reports pelvis and
upper-body rates in three windows after the flip, against the BWC reference,
which stays still for 1 s. FLIP_DELAY_MS adds a fixed actuator delay.

  uv run python scripts/tools/flip_probe.py <run> <model.pt>
"""
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
N = 16
torch.manual_seed(0)
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = N
for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
  cfg.terminations.pop(k, None)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v2.npz"
# FLIP_DELAY_MS: fixed actuator delay, as in delay_probe.py.
_lag = int(round(float(__import__("os").environ.get("FLIP_DELAY_MS", "0")) / 1000
                 / cfg.sim.mujoco.timestep))
for _a in cfg.scene.entities["robot"].articulation.actuators:
  _a.delay_min_lag = _lag
  _a.delay_max_lag = _lag
CMD = torch.zeros(N, 3)
# RHPS1_CMD_TAU: the policy sees the command through the controller's filter.
TAU = float(__import__("os").environ.get("RHPS1_CMD_TAU", "0"))
_filt = {"y": torch.zeros(N, 3), "step": -1}


def _cmd(env, command_name, tau=0.0):
  if TAU <= 0:
    return CMD.to(env.device)
  step = int(env.common_step_counter)
  if _filt["step"] != step:
    _filt["y"] = _filt["y"] + (env.step_dt / TAU) * (CMD - _filt["y"])
    _filt["step"] = step
  return _filt["y"].to(env.device)


cfg.observations["actor"].terms["velocity_command"].func = _cmd
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True},
            strict=True, map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
robot = u.scene["robot"]
sens = u.scene["feet_ground_contact"]
names = list(robot.joint_names)
upper = [i for i, n in enumerate(names) if not any(t in n for t in ("CROTCH", "KNEE", "ANKLE"))]
dt = u.step_dt

obs, _ = env.reset()
q0 = robot.data.default_joint_pos.clone()
# Envs start apart (+-0.03 rad, +-0.1 m/s), otherwise they are one sample repeated.
robot.write_joint_state_to_sim(q0 + (torch.rand_like(q0) * 2 - 1) * 0.03, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
root[:, 7:9] = (torch.rand(N, 2) * 2 - 1) * 0.1
robot.write_root_state_to_sim(root)
W, F, QU, X = [], [], [], []
# Each env flips after its own 2-5 s of standing, so the flips land on different
# states; a settled stance otherwise makes every env the same sample.
flip = (torch.rand(N) * 3.0 + 2.0) / dt
flip = flip.long().numpy()
with torch.no_grad():
  for _ in range(int(3.0 / dt)):
    obs, _, _, _ = env.step(torch.zeros(N, env.num_actions))
  for k in range(int(10.0 / dt)):
    CMD[:, 0] = torch.as_tensor(k >= flip, dtype=torch.float32)
    obs, _, _, _ = env.step(pol(obs))
    W.append(robot.data.root_link_ang_vel_b.norm(dim=-1).numpy().copy())
    F.append(sens.data.force.norm(dim=-1).squeeze(-1).numpy().copy())
    QU.append(robot.data.joint_vel[:, upper].abs().max(dim=-1).values.numpy().copy())
    X.append(robot.data.root_link_pos_w[:, 0].numpy().copy())
W, F, QU, X = np.stack(W), np.stack(F), np.stack(QU), np.stack(X)
D = np.degrees


def win(x, a, b):
  """Per-env max over [a, b) s after that env's flip."""
  return np.array([x[flip[e] + int(a / dt):flip[e] + int(b / dt), e].max() for e in range(N)])


unload = []
for e in range(N):
  off = F[flip[e]:, e].min(-1) < 5.0
  unload.append(np.argmax(off) * dt if off.any() else np.nan)
print(f"{run} {ck}  retard {_lag * cfg.sim.mujoco.timestep * 1000:.0f} ms, bascule 0 -> avant ({N} departs, median / p90)")
print("  (reference BWC : immobile 0-1 s, bassin 8 deg/s a 1-2.3 s, pied leve a 2.3 s)")
for a, b in ((0.0, 0.3), (0.3, 1.0), (1.0, 2.3)):
  w, q = D(win(W, a, b)), D(win(QU, a, b))
  print(f"  {a:.1f}-{b:.1f} s  bassin {np.median(w):5.1f} / {np.percentile(w, 90):5.1f} deg/s"
        f"   haut du corps {np.median(q):5.1f} / {np.percentile(q, 90):5.1f} deg/s")
print(f"  premier pied decharge {np.nanmedian(unload):5.2f} s apres la bascule (median)")
dx = np.array([X[min(flip[e] + int(5.0 / dt), len(X) - 1), e] - X[flip[e], e] for e in range(N)])
print(f"  avance 5 s apres la bascule {np.median(dx):5.2f} m (median ; BWC 0.42 m)")
