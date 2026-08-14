"""Does the base_lin_vel bias decide whether the robot walks?

plant_sweep found the only two environments that really walked both drew a
base_lin_vel x bias near -0.045, the extreme of the (-0.05, 0.05) range, while
no other parameter separated anything.

A negative bias tells the policy it is slower than it really is, so it pushes
harder. At zero bias it believes it already tracks the command. That matters for
deployment: mc_mujoco injects no bias at all, and the real robot's floating base
observer was measured at +0.0167 m/s while walking -- the opposite sign.

Here the plant is nominal and the ONLY thing that varies is that bias, set by
hand per environment across the range.

  uv run python scripts/tools/vx_bias_sweep.py <checkpoint.pt> [speed]
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
DURATION_S = 6.0
POLICY_DT = 0.005
WARMUP = 200

PLANT_EVENTS = (
  "link_inertia", "link_com", "base_com", "encoder_bias",
  "foot_friction", "actuator_gains", "posture_filter", "sensor_bias",
  "push_base",
)
BIASES = [-0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.0167, 0.03, 0.05]
REPEATS = 8


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  n = len(BIASES) * REPEATS

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = n
  cfg.curriculum = {}
  for name in PLANT_EVENTS:
    cfg.events.pop(name, None)

  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")
  target = torch.tensor([fwd, 0.0, 0.0], device=device)

  def forced_update() -> None:
    cmd.vel_command_b[:] = target
    cmd.vel_command_w[:] = target
    cmd.vel_command_out[:] = target

  cmd._update_command = forced_update  # type: ignore[method-assign]
  cmd.is_standing_env[:] = False

  env.reset()
  cmd.is_standing_env[:] = False

  # sensor_bias was removed, so install the store by hand: observations read it
  # through getattr, and a missing key simply means no bias.
  bias = torch.zeros((n, 3), device=device)
  for i, b in enumerate(BIASES):
    bias[i * REPEATS : (i + 1) * REPEATS, 0] = b
  setattr(env_raw, "_rhps1_sensor_bias", {"base_lin_vel": bias})

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  prev = robot.data.root_link_pos_w[:, :2].clone()
  path = torch.zeros(n, device=device)
  vx = []
  for i in range(int(DURATION_S / POLICY_DT)):
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    xy = robot.data.root_link_pos_w[:, :2]
    if i >= WARMUP:
      path += torch.norm(xy - prev, dim=-1)
      vx.append(robot.data.root_link_lin_vel_b[:, 0].clone())
    prev = xy.clone()

  v = torch.stack(vx).mean(dim=0).cpu().numpy()
  p = path.cpu().numpy()
  expected = fwd * (DURATION_S - WARMUP * POLICY_DT)

  print(f"\nplant NOMINAL, commande {fwd} m/s, {REPEATS} tirages par biais")
  print(f"chemin attendu si le robot suit : {expected:.2f} m\n")
  print(f"{'biais v_x':>10s} {'chemin median':>14s} {'chemin max':>11s} {'v_x median':>11s}")
  for i, b in enumerate(BIASES):
    sl = slice(i * REPEATS, (i + 1) * REPEATS)
    tag = "  <- robot reel" if abs(b - 0.0167) < 1e-6 else ("  <- mc_mujoco" if b == 0.0 else "")
    print(f"{b:+10.4f} {np.median(p[sl]):13.3f}m {p[sl].max():10.3f}m "
          f"{np.median(v[sl]):+11.3f}{tag}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
