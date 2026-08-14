"""Does the initial joint offset decide whether the robot walks?

plant_sweep showed the drawn plant does not: no randomised parameter separates
the fastest environments from the rest, yet the distribution is heavy tailed --
median 5 cm walked against a max of 1.84 m.

What is left that still differs per environment is the initial posture,
reset_robot_joints at +/-0.05 rad. Here the plant is frozen (every randomisation
event removed) so the ONLY difference between environments is that offset, drawn
over a range given on the command line.

Env 0 always starts exactly nominal, as a control: it is what play and mc_mujoco
do.

  uv run python scripts/tools/initial_pose_sweep.py <checkpoint.pt> [speed] [range]
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
NUM_ENVS = 256
WARMUP = 200

PLANT_EVENTS = (
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


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  spread = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}
  for name in PLANT_EVENTS:
    cfg.events.pop(name, None)
  cfg.events["reset_robot_joints"].params["position_range"] = (-spread, spread)

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

  # Env 0 is the control: exactly the nominal posture, like play and mc_mujoco.
  q = robot.data.joint_pos.clone()
  q[0] = robot.data.default_joint_pos[0]
  robot.write_joint_position_to_sim(q)
  offset = (q - robot.data.default_joint_pos).abs().mean(dim=-1).cpu().numpy()

  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  prev = robot.data.root_link_pos_w[:, :2].clone()
  path = torch.zeros(NUM_ENVS, device=device)
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

  print(f"\nplant FIGE, offset initial +/-{spread} rad, commande {fwd} m/s")
  print(f"v_x    p10 {np.percentile(v, 10):+.3f}  p50 {np.percentile(v, 50):+.3f}"
        f"  p90 {np.percentile(v, 90):+.3f}  max {v.max():+.3f}")
  print(f"chemin p50 {np.percentile(p, 50):.3f}  p90 {np.percentile(p, 90):.3f}"
        f"  max {p.max():.3f} m   (attendu {expected:.2f} m)")
  print(f"envs au-dessus de 50% du chemin : {int((p > 0.5 * expected).sum())}/{NUM_ENVS}")
  print(f"\nenv 0 (posture nominale, = play/mc_mujoco) : v_x {v[0]:+.3f}"
        f"  chemin {p[0]:.3f} m")
  corr = float(np.corrcoef(offset, v)[0, 1])
  print(f"correlation offset initial / v_x : {corr:+.2f}")
  top = np.argsort(-v)[: max(int(0.15 * NUM_ENVS), 5)]
  print(f"offset moyen des 15% meilleurs {offset[top].mean():.4f} rad"
        f"  contre {offset.mean():.4f} sur tous")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
