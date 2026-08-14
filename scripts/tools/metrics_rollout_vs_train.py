"""Read the same metrics in a rollout as the training loop logs.

Compares outputs rather than configurations: whichever metric collapses points
at the cause.

  uv run python scripts/tools/metrics_rollout_vs_train.py <checkpoint.pt> [--play]
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-RHPS1"
DURATION_S = 10.0
POLICY_DT = 0.005
NUM_ENVS = 256
WARMUP_STEPS = 200

# Valeurs loguees par le run d'origine a l'iteration 14999.
TRAIN_REF = {
  "air_time_mean": 0.3293,
  "double_flight_rate": 0.0072,
  "foot_vel_max": 1.8088,
  "landing_vel_mean": 0.0832,
  "left_foot_marker_speed": 0.2018,
  "loaded_foot_fraction": 0.6567,
  "peak_height_mean": 0.0013,
  "slip_velocity_mean": 0.0275,
  "sole_height_max": 0.1873,
  "sole_height_p50": 0.0009,
  "sole_height_p90": 0.0416,
  "sole_height_p99": 0.0618,
  "stance_contacts_mean": 2.8660,
  "torque_limit_ratio_mean": 0.9126,
}


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  play = "--play" in sys.argv
  # Par defaut on ne force PAS la commande : on veut la distribution
  # d'entrainement, pas un cas particulier.
  force = "--force-cmd" in sys.argv

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=play)
  cfg.scene.num_envs = NUM_ENVS
  cfg.episode_length_s = max(cfg.episode_length_s, 2 * DURATION_S)

  env_raw = ManagerBasedRlEnv(cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  mm = env_raw.metrics_manager
  names = list(mm._term_names)  # noqa: SLF001

  if force:
    cmd = env_raw.command_manager.get_term("twist")
    target = torch.tensor([0.2, 0.0, 0.0], device=device)

    def forced_update() -> None:
      cmd.vel_command_b[:] = target
      cmd.vel_command_w[:] = target
      cmd.vel_command_out[:] = target

    cmd._update_command = forced_update  # type: ignore[method-assign]

  env.reset()
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  acc = torch.zeros(len(names), device=device)
  n = 0
  for i in range(int(DURATION_S / POLICY_DT)):
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    if i >= WARMUP_STEPS:
      acc += mm._step_values.mean(dim=0)  # noqa: SLF001
      n += 1

  got = (acc / max(n, 1)).cpu().numpy()

  mode = "play" if play else "entrainement"
  cmdmode = "commande forcee 0.2" if force else "commandes naturelles"
  print(f"\nrollout : config {mode}, {cmdmode}, {NUM_ENVS} envs, deterministe\n")
  print(f"{'metrique':32s} {'rollout':>10s} {'train 14999':>12s} {'rapport':>9s}")
  for name, v in sorted(zip(names, got, strict=True)):
    # Tensorboard tags suffix the term name (_mean, _p90, ...), so match on
    # the prefix.
    ref = TRAIN_REF.get(name)
    if ref is None:
      cands = [k for k in TRAIN_REF if k.startswith(name)]
      ref = TRAIN_REF[cands[0]] if len(cands) == 1 else None
    if ref is None:
      print(f"{name:32s} {v:+10.4f} {'-':>12s} {'-':>9s}")
      continue
    ratio = f"{v / ref:8.2f}x" if abs(ref) > 1e-9 else "       -"
    print(f"{name:32s} {v:+10.4f} {ref:+12.4f} {ratio:>9s}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
