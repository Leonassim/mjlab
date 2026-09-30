"""Command onset while standing in control: the operator's joystick push.

Settle 3 s with actions held, policy in control at command 0 for 4 s, then the
command flips to forward. Reports the jolt in the 0.3 s after the flip and the
delay to the first step. Arming via the controller now starts the policy at its
own stance, so the 4 s at command 0 stand in for that.

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
N = 8
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
CMD = torch.zeros(3)
cfg.observations["actor"].terms["velocity_command"].func = (
  lambda env, command_name: CMD.to(env.device).repeat(env.num_envs, 1)
)
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
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += u.scene.env_origins
robot.write_root_state_to_sim(root)
W, F, QU, AIR = [], [], [], []
with torch.no_grad():
  for _ in range(int(3.0 / dt)):
    obs, _, _, _ = env.step(torch.zeros(N, env.num_actions))
  for _ in range(int(4.0 / dt)):
    obs, _, _, _ = env.step(pol(obs))
  CMD[0] = 1.0
  for _ in range(int(3.0 / dt)):
    obs, _, _, _ = env.step(pol(obs))
    W.append(robot.data.root_link_ang_vel_b.norm(dim=-1).numpy().copy())
    f = sens.data.force.norm(dim=-1).squeeze(-1).numpy().copy()
    F.append(f)
    QU.append(robot.data.joint_vel[:, upper].abs().max(dim=-1).values.numpy().copy())
W, F, QU = np.stack(W), np.stack(F), np.stack(QU)
n3 = int(0.3 / dt)
share = F / np.clip(F.sum(-1, keepdims=True), 1e-6, None)
dshare = np.abs(np.diff(share[: n3 + 1, :, 0], axis=0)).max(0) / dt  # load transfer rate
first_step = [(np.argmax(F[:, e].min(-1) < 5.0) * dt) if (F[:, e].min(-1) < 5.0).any() else np.nan
              for e in range(N)]
D = np.degrees
print(f"{run} {ck}  retard {_lag * cfg.sim.mujoco.timestep * 1000:.0f} ms, bascule 0 -> avant, 0.3 s qui suivent :")
print(f"  rotation bassin max   {D(W[:n3].max(0)).mean():5.1f} deg/s (median sur {N})")
print(f"  haut du corps max     {D(QU[:n3].max(0)).mean():5.1f} deg/s")
print(f"  transfert de charge   {np.median(dshare) * 100:5.0f} %/s du poids")
print(f"  premier pied decharge {np.nanmedian(first_step):5.2f} s apres la bascule")
