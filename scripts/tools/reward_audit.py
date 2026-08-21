"""What the corrected rewards say about gaits we already know.

Parity asks "what weight keeps July's pressure?". That is the wrong target: July's
pressure is what produced policy 0's flat 2 cm swing. This asks the other
question -- given a walking policy and a broken one, does the reward rank them
correctly? A term that charges the broken gait less than policy 0 is a bad
objective and no weight will fix it.

  uv run python scripts/tools/reward_audit.py

One env, several policies, identical conditions. Prints gait statistics and the
raw cost of every reward variant per policy, plus the discrimination margin.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import os
from pathlib import Path

import torch

L = "logs/rsl_rl/rhps1_velocity"

# label -> checkpoint. Two that walk, two that broke in opposite ways.
POLICIES: list[tuple[str, str]] = [
  ("policy0", f"{L}/2026-07-10_20-59-17/model_9900.pt"),
  ("p0+rand", f"{L}/2026-08-19_23-39-10/model_700.pt"),
  ("+fs@-11", f"{L}/2026-08-20_10-24-39/model_700.pt"),
  ("+fs@-3.2", f"{L}/2026-08-20_11-33-59/model_700.pt"),
]

# term -> {variant: params}. First variant is July's.
VARIANTS: dict[str, dict[str, dict]] = {
  "flat_support": {
    "july": {"corner_tolerance": 0.0, "change_gain": 0.0,
             "standing_threshold": -1.0, "load_threshold": 0.0},
    "corner": {"corner_tolerance": 0.001, "change_gain": 0.0,
               "standing_threshold": -1.0, "load_threshold": 0.0},
    "change": {"corner_tolerance": 0.0, "change_gain": 1.0,
               "standing_threshold": -1.0, "load_threshold": 0.0},
    "load": {"corner_tolerance": 0.0, "change_gain": 0.0,
             "standing_threshold": 0.1, "load_threshold": 140.0},
    "all": {"corner_tolerance": 0.001, "change_gain": 1.0,
            "standing_threshold": 0.1, "load_threshold": 140.0},
  },
  "air_time": {
    "july": {"power": 2.0, "touchdown_cost": 0.15},
    "linear": {"power": 1.0, "touchdown_cost": 0.0},
    "lin+cost": {"power": 1.0, "touchdown_cost": 0.15},
  },
  "min_foot_height": {
    "july": {"min_height": 0.08},
    "h04": {"min_height": 0.04},
    "h02": {"min_height": 0.02},
  },
  # The other height term, and the one with the right shape: peak per swing,
  # charged once at landing. Its target of 0.15 is the question -- squared
  # relative error saturates near 1.0 when the policy is two orders below it.
  "foot_swing_height": {
    "t15": {"target_height": 0.15},
    "t08": {"target_height": 0.08},
    "t04": {"target_height": 0.04},
    "t02": {"target_height": 0.02},
  },
  "impact_vel": {"july": {"limit": 0.1}, "corrected": {"limit": 0.15}},
}


def pct(t: torch.Tensor, q: float) -> float:
  return float(torch.quantile(t.flatten().float(), q))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", type=int, default=400)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--ablation", default="p0+rand")
  args = ap.parse_args()
  os.environ.setdefault("RHPS1_ABLATION", args.ablation)

  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  task, device = "Mjlab-Velocity-Flat-RHPS1", "cuda:0"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)

  # One live instance per variant. Reusing term_cfg.func would share state --
  # air_time and flat_support both carry it, so a variant would measure itself.
  live: dict[tuple[str, str], tuple] = {}
  for term, variants in VARIANTS.items():
    base = env.reward_manager.get_term_cfg(term)
    for vname, params in variants.items():
      c = copy.deepcopy(base)
      c.params = {**base.params, **params}
      f = base.func
      live[(term, vname)] = (
        f if inspect.isfunction(f) else type(f)(c, env), c.params)

  robot = env.scene["robot"]
  # The reward manager already resolved site names to ids; EntityData exposes no
  # name lookup of its own.
  feet_ids = env.reward_manager.get_term_cfg(
    "min_foot_height").params["asset_cfg"].site_ids
  contact = env.scene["feet_ground_contact"]

  runner = load_runner_cls(task)(wrapped, asdict(load_rl_cfg(task)), device=device)
  out: dict[str, dict] = {}

  for label, ckpt in POLICIES:
    if not Path(ckpt).exists():
      print(f"skip {label}: {ckpt} missing")
      continue
    runner.load(ckpt, map_location=device)
    policy = runner.get_inference_policy(device=device)
    obs, _ = env.reset()
    all_ids = torch.arange(env.num_envs, device=device)
    for f, _ in live.values():
      if hasattr(f, "reset"):
        f.reset(all_ids)

    costs = {k: 0.0 for k in live}
    swing_z, air_frac, stance, torque, prog, land_t = [], [], [], [], [], []
    with torch.inference_mode():
      for i in range(args.steps):
        obs, _, _, _, _ = env.step(policy(obs))
        if i < 50:  # settle
          continue
        for k, (f, p) in live.items():
          costs[k] += float(f(env, **p).mean())
        z = robot.data.site_pos_w[:, feet_ids, 2]
        in_air = contact.data.found == 0
        if in_air.any():
          swing_z.append(z[in_air].detach().clone())
        air_frac.append(in_air.float().mean())
        fc = contact.data.last_air_time
        if fc is not None:
          landed = contact.compute_first_contact(dt=env.step_dt)
          if landed.any():
            land_t.append(fc[landed].detach().clone())
        stance.append((contact.data.found > 0).float().sum(dim=1).mean())
        try:
          torque.append(
            (robot.data.actuator_force.abs()
             / robot.data.actuator_force_limit.clamp(min=1e-6)).mean())
        except AttributeError:
          pass
        try:
          prog.append(robot.data.root_link_lin_vel_b[:, 0].abs().mean())
        except AttributeError:
          pass

    n = args.steps - 50
    sz = torch.cat(swing_z) if swing_z else torch.zeros(1, device=device)
    out[label] = {
      "cost": {k: v / n for k, v in costs.items()},
      "swing_p50": pct(sz, 0.5), "swing_p90": pct(sz, 0.9),
      "swing_max": float(sz.max()),
      "air_frac": float(torch.stack(air_frac).mean()),
      "land_t_p50": pct(torch.cat(land_t), 0.5) if land_t else float("nan"),
      "land_t_p90": pct(torch.cat(land_t), 0.9) if land_t else float("nan"),
      "stance": float(torch.stack(stance).mean()),
      "torque": float(torch.stack(torque).mean()) if torque else float("nan"),
      "speed": float(torch.stack(prog).mean()) if prog else float("nan"),
    }
    print(f"done {label}")

  labels = list(out)
  w = 11
  print(f"\n=== gait, {args.steps - 50} steps, {args.num_envs} envs, "
        f"ablation={os.environ['RHPS1_ABLATION']} ===\n")
  print(f"{'':<22}" + "".join(f"{k:>{w}}" for k in labels))
  for stat in ("swing_p50", "swing_p90", "swing_max", "air_frac",
               "land_t_p50", "land_t_p90", "stance", "torque", "speed"):
    print(f"{stat:<22}" + "".join(f"{out[k][stat]:>{w}.4f}" for k in labels))

  good = [k for k in labels if k in ("policy0", "p0+rand")]
  bad = [k for k in labels if k not in good]
  print(f"\n=== raw cost per variant (before weight) ===")
  print(f"{'term / variant':<22}" + "".join(f"{k:>{w}}" for k in labels)
        + f"{'bad-good':>11}")
  for term, variants in VARIANTS.items():
    for vname in variants:
      key = (term, vname)
      row = [out[k]["cost"][key] for k in labels]
      d = (sum(out[k]["cost"][key] for k in bad) / max(len(bad), 1)
           - sum(out[k]["cost"][key] for k in good) / max(len(good), 1))
      print(f"{term[:14] + '/' + vname:<22}"
            + "".join(f"{v:>{w}.4f}" for v in row) + f"{d:>11.4f}")
  print("\nbad-good > 0 means the penalty charges the broken gaits more, i.e. the "
        "term\nranks them correctly. <= 0 means no weight can make it useful.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
