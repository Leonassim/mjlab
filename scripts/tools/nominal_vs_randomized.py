"""Does the policy walk on the NOMINAL robot, the one that gets deployed?

rsl_rl has no evaluation phase, so nothing during training ever tests the robot
without randomisation -- while mc_mujoco and the real robot are exactly that.

Measured in BODY frame velocity and path length, not world x displacement:
with yaw the robot walks a curve and its world x understates the walk.

  uv run python scripts/tools/nominal_vs_randomized.py <checkpoint.pt> [speed]
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
NUM_ENVS = 64
WARMUP = 200

# Events that define the plant. reset_base and reset_robot_joints stay: they
# place the robot, they do not modify it.
DR_EVENTS = (
  "link_inertia",
  "link_com",
  "base_com",
  "encoder_bias",
  "foot_friction",
  "actuator_gains",
  "posture_filter",
  "sensor_bias",
  "push_base",
)


def run(
  ckpt: str, fwd: float, nominal: bool, play: bool = True, stochastic: bool = False
) -> tuple[float, float, float]:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=play)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}
  if nominal:
    for name in DR_EVENTS:
      cfg.events.pop(name, None)
    # Et la pose de depart exactement nominale, sans dispersion.
    if "reset_robot_joints" in cfg.events:
      cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)

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
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  prev = robot.data.root_link_pos_w[:, :2].clone()
  path = torch.zeros(NUM_ENVS, device=device)
  vx = []
  for i in range(int(DURATION_S / POLICY_DT)):
    with torch.inference_mode():
      action = policy(obs, stochastic_output=True) if stochastic else policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    xy = robot.data.root_link_pos_w[:, :2]
    if i >= WARMUP:
      path += torch.norm(xy - prev, dim=-1)
      vx.append(robot.data.root_link_lin_vel_b[:, 0].clone())
    prev = xy.clone()

  v = torch.stack(vx).mean(dim=0).cpu().numpy()
  p = path.cpu().numpy()
  env_raw.close()
  return float(np.percentile(v, 50)), float(np.percentile(v, 90)), float(
    np.percentile(p, 50)
  )


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  window = DURATION_S - WARMUP * POLICY_DT

  print(f"commande {fwd} m/s en avant, lacet nul, fenetre utile {window:.1f} s")
  print(f"chemin attendu si le robot suit : {fwd * window:.2f} m\n")
  conditions = [
    ("nominal, deploiement", True, True, False),
    ("randomise, deploiement", False, True, False),
    ("randomise, train det", False, False, False),
    ("randomise, train stoch", False, False, True),
  ]
  print(f"{'condition':26s} {'v_x p50':>9s} {'v_x p90':>9s} {'chemin p50':>11s}")
  for label, nominal, play, stoch in conditions:
    v50, v90, p50 = run(ckpt, fwd, nominal, play=play, stochastic=stoch)
    print(f"{label:26s} {v50:+8.3f} {v90:+8.3f} {p50:10.3f}m", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
