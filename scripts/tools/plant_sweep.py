"""Which drawn plant walks, and which parameter decides?

Domain randomisation gives every environment a different robot. If only some of
them advance, the parameter that separates the winners from the rest is the one
to tighten in the next run.

Deployment conditions: play config (no observation noise), deterministic policy,
command injected in _update_command. Per environment we log the drawn plant and
the realised forward speed, then rank the parameters by how well they separate.

  uv run python scripts/tools/plant_sweep.py <checkpoint.pt> [speed]
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


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.curriculum = {}

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

  # Plant drawn for each env, read after the reset that drew it.
  feats: dict[str, np.ndarray] = {}
  md = env_raw.sim.model
  mass = md.body_mass
  if mass is not None and mass.dim() > 1:
    feats["masse_kg"] = mass.sum(dim=-1).float().cpu().numpy()
  fr = md.geom_friction
  if fr is not None and fr.dim() >= 2:
    # The randomised geoms are the eight foot patches: take the lowest, which is
    # what limits traction.
    feats["friction_min"] = fr[..., 0].min(dim=-1).values.float().cpu().numpy()
  for act in robot.actuators:
    ps = getattr(act, "posture_stiffness", None)
    if ps is not None:
      feats["posture_K"] = ps.float().mean(dim=-1).cpu().numpy()
    st = getattr(act, "stiffness", None)
    dk = getattr(act, "default_stiffness", None)
    if st is not None and dk is not None:
      feats["gain_kp_ratio"] = (st / dk).float().mean(dim=-1).cpu().numpy()
    break
  eb = getattr(robot.data, "encoder_bias", None)
  if eb is not None:
    feats["encoder_bias_abs"] = eb.abs().float().mean(dim=-1).cpu().numpy()
  sb = getattr(env_raw, "_rhps1_sensor_bias", {})
  if "base_lin_vel" in sb:
    feats["biais_vx"] = sb["base_lin_vel"][:, 0].float().cpu().numpy()
  if "projected_gravity" in sb:
    feats["biais_grav_x"] = sb["projected_gravity"][:, 0].float().cpu().numpy()

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

  print(f"\ncommande {fwd} m/s, {NUM_ENVS} tirages, deterministe")
  print(f"v_x   p10 {np.percentile(v, 10):+.3f}  p50 {np.percentile(v, 50):+.3f} "
        f" p90 {np.percentile(v, 90):+.3f}  max {v.max():+.3f}")
  print(f"chemin p50 {np.percentile(p, 50):.3f} m  p90 {np.percentile(p, 90):.3f} m "
        f" max {p.max():.3f} m   (attendu {fwd * (DURATION_S - WARMUP * POLICY_DT):.2f} m)")

  k = max(int(0.15 * NUM_ENVS), 5)
  best = np.argsort(-v)[:k]
  worst = np.argsort(v)[:k]
  print(f"\n{'parametre':18s} {'meilleurs 15%':>14s} {'pires 15%':>12s} "
        f"{'ecart':>8s} {'correlation':>12s}")
  rows = []
  for name, arr in feats.items():
    if arr.shape[0] != NUM_ENVS:
      continue
    mb, mw = float(arr[best].mean()), float(arr[worst].mean())
    sd = float(arr.std()) or 1.0
    corr = float(np.corrcoef(arr, v)[0, 1])
    rows.append((abs(corr), name, mb, mw, (mb - mw) / sd, corr))
  for _, name, mb, mw, gap, corr in sorted(rows, reverse=True):
    print(f"{name:18s} {mb:14.4f} {mw:12.4f} {gap:+8.2f}s {corr:+12.2f}")
  print("\nEcart en ecarts-types. Une correlation sous ~0.15 n'est pas separante.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
