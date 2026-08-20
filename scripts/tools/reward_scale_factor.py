"""How much a reward correction rescales its own cost, on identical trajectories.

A correction changes what is measured, so the weight in front of it no longer
means what it did. Reading the ratio off two training runs does not settle it --
they learned different gaits, so measurement and behaviour move together. This
rolls out one checkpoint and evaluates both parameterisations on the same states.

  uv run python scripts/tools/reward_scale_factor.py logs/.../model_700.pt

Prints, per term, the raw-cost ratio corrected/July and the weight that restores
the realized value: parity_weight = old_weight / ratio.

Rollouts read absolute magnitudes low against the training loop; this is a ratio
between two variants on the same trajectories, so that bias cancels.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import os
from pathlib import Path

import torch

# July parameterisation against today's, per term. Keys are reward term names;
# values are (july_params, corrected_params, july_weight, corrected_weight).
VARIANTS: dict[str, tuple[dict, dict, float, float]] = {
  "flat_support": (
    {"corner_tolerance": 0.0, "change_gain": 0.0,
     "standing_threshold": -1.0, "load_threshold": 0.0},
    {"corner_tolerance": 0.001, "change_gain": 1.0,
     "standing_threshold": 0.1, "load_threshold": 140.0},
    -2.4, -11.0,
  ),
  "air_time": (
    {"power": 2.0, "touchdown_cost": 0.15},
    {"power": 1.0, "touchdown_cost": 0.0},
    2.0, 2.0,
  ),
  "standing_single_support": (
    {"grace_period": 0.0}, {"grace_period": 1.5}, -4.0, -6.0,
  ),
  "impact_vel": ({"limit": 0.1}, {"limit": 0.15}, -0.5, -2.0),
}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint", type=Path)
  ap.add_argument("--steps", type=int, default=400)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--ablation", default="p0+rand")
  ap.add_argument(
    "--standing-frac", type=float, default=None,
    help="override rel_standing_envs. Needed for standing_single_support: its "
         "grace_period only fires on a transition into the standing regime, and "
         "a default rollout has too few to measure. Use 1.0.",
  )
  args = ap.parse_args()

  os.environ.setdefault("RHPS1_ABLATION", args.ablation)

  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  task = "Mjlab-Velocity-Flat-RHPS1"
  device = "cuda:0"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = args.num_envs
  if args.standing_frac is not None:
    # Set on the config, before the env is built. Forcing a command after
    # env.step() does nothing -- the command term overwrites it on resample.
    cfg.commands["twist"].rel_standing_envs = args.standing_frac
    cfg.curriculum.pop("standing_envs", None)  # else the curriculum puts it back
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)  # the runner needs num_actions from it

  # One live term object per variant, sharing the env so both see the same state.
  live: dict[str, tuple] = {}
  for name, (july, corrected, _, _) in VARIANTS.items():
    term_cfg = env.reward_manager.get_term_cfg(name)
    func = term_cfg.func
    def build(params):
      c = copy.deepcopy(term_cfg)
      c.params = {**term_cfg.params, **params}
      # term_cfg.func is the *instance* the reward manager built, not the class,
      # so isinstance(func, type) is False and reusing it would make both
      # variants share one object -- and one grace_left, one prev_count. Any
      # term whose correction acts through state would then measure itself.
      inst = func if inspect.isfunction(func) else type(func)(c, env)
      return inst, c.params
    live[name] = (build(july), build(corrected))

  # A real policy, not zero actions: the ratios are only meaningful on the
  # trajectories the reward is supposed to shape. A collapsing robot never
  # triggers standing_single_support's grace, for one.
  runner = load_runner_cls(task)(wrapped, asdict(load_rl_cfg(task)), device=device)
  runner.load(str(args.checkpoint), map_location=device)
  policy = runner.get_inference_policy(device=device)
  obs, _ = env.reset()

  totals = {n: [0.0, 0.0] for n in VARIANTS}
  count = 0
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, _, _, _ = env.step(policy(obs))
      for name, ((f_j, p_j), (f_c, p_c)) in live.items():
        totals[name][0] += float(f_j(env, **p_j).mean())
        totals[name][1] += float(f_c(env, **p_c).mean())
      count += 1

  print(f"\n{count} steps, {args.num_envs} envs, ablation={os.environ['RHPS1_ABLATION']}\n")
  print(f"{'term':<26}{'cost July':>12}{'cost corr.':>12}{'ratio':>9}"
        f"{'weight now':>12}{'parity':>10}")
  for name, (a, b) in totals.items():
    a, b = a / count, b / count
    _, _, w_july, w_now = VARIANTS[name]
    ratio = b / a if a else float("nan")
    parity = w_july / ratio if ratio else float("nan")
    print(f"{name[:25]:<26}{a:12.5f}{b:12.5f}{ratio:9.3f}{w_now:12.2f}{parity:10.2f}")
  print("\nparity = the weight at which the corrected term contributes what the "
        "July one did.\nRun it, then sweep 0.5x and 2x if the verdict is ambiguous.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
