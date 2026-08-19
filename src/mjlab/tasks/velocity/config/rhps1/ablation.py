"""Ablation ladder from policy 0.

Policy 0 (run 2026-07-10_13-52-54) is the only configuration that ever produced
a walking gait on hardware. Everything added since is a deviation, and three
consecutive runs carrying all of them at once failed to walk while policy 0 was
already single-support by iteration 600. Guessing which deviation costs the gait
has a poor record, so this selects them one at a time instead.

  RHPS1_ABLATION=p0              policy 0, every deviation reverted
  RHPS1_ABLATION=p0+obs246       ... plus the 5-step action history
  RHPS1_ABLATION=p0+obs246+knee  ... several, applied in order

Unset leaves the configuration untouched, so nothing changes for a normal run.

The discriminator is Metrics/stance_contacts_mean at iteration ~600: 4 means
both feet planted, policy 0 sat at 2.33. One hour per rung at 609 it/h.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

ENV_VAR = "RHPS1_ABLATION"


def selection() -> list[str] | None:
  """Steps requested, or None when the variable is unset."""
  raw = os.environ.get(ENV_VAR, "").strip()
  if not raw:
    return None
  steps = [s for s in raw.split("+") if s]
  if not steps or steps[0] != "p0":
    raise ValueError(f"{ENV_VAR} must start with 'p0', got {raw!r}")
  unknown = set(steps[1:]) - set(DELTAS)
  if unknown:
    raise ValueError(f"{ENV_VAR}: unknown steps {sorted(unknown)}, have {sorted(DELTAS)}")
  return steps


def mirror_enabled() -> bool:
  """Mirror loss is a deviation: policy 0 trained without it."""
  steps = selection()
  return True if steps is None else "mirror" in steps


# --------------------------------------------------------------------------
# Baseline


def _revert_to_policy0(cfg: ManagerBasedRlEnvCfg) -> None:
  r = cfg.rewards

  r["angular_momentum"].weight = -0.2
  r["flat_support"].weight = -2.4
  r["flat_support"].params.pop("command_name", None)
  r["flat_support"].params.pop("corner_tolerance", None)
  r["foot_clearance"].weight = -4.0
  r["foot_slip"].weight = -0.3
  r["impact_vel"].weight = -0.5
  r["impact_vel"].params["limit"] = 0.1
  r["standing_single_support"].weight = -4.0
  r["air_time"].params.pop("power", None)
  r["air_time"].params.pop("touchdown_cost", None)

  # Terms policy 0 carried that were dropped since. min_foot_height is the one
  # that paid zero for standing and punished a short lift -- restored anyway:
  # the baseline reproduces policy 0, defects included, because that is the
  # configuration that walked.
  r["min_foot_height"] = RewardTermCfg(
    func=mdp.swing_foot_height, weight=-5.0,
    params={"min_height": 0.08, "sensor_name": "feet_ground_contact",
            "command_name": "twist", "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
  )
  r["flat_touchdown"] = RewardTermCfg(
    func=mdp.flat_touchdown_penalty, weight=-1.8,
    params={"sensor_name": "feet_ground_contact_split", "required_contacts_per_foot": 4,
            "command_name": "twist", "command_threshold": 0.05},
  )
  r["ankle_roll_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2, weight=-0.002,
    params={"actuator_pattern": r"^[LR]_ANKLE_R$"},
  )

  for name in PROXIMITY_TERMS:
    r.pop(name, None)

  for name in EXTRA_EVENTS:
    cfg.events.pop(name, None)
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)

  cfg.observations["actor"].terms["actions"].history_length = 0

  # Knee effort limit 100 N.m rather than 70, which is what the scale encodes.
  # 0.0075, read from policy 0's params/env.yaml -- its ONNX metadata rounds to
  # three decimals and reports 0.007, which is not the value.
  for j in ("L_KNEE_P", "R_KNEE_P"):
    cfg.actions["joint_pos"].scale[j] = 0.0075

  # Zero weight, so inert, but it keeps the baseline diff against policy 0 empty.
  cfg.rewards["action_acc_l2"] = RewardTermCfg(func=mdp.action_acc_l2, weight=0.0)


EXTRA_EVENTS = ("actuator_gains", "link_com", "link_inertia", "sensor_bias")

PROXIMITY_TERMS = (
  "arm_torso_proximity", "knee_proximity", "leg_proximity",
  "shoulder_body_proximity", "shoulder_chest_proximity", "wrist_thigh_proximity",
)


# --------------------------------------------------------------------------
# Rungs. Each is baseline + this one deviation, so the cost of a change is
# read directly rather than inferred from a run carrying all of them.


def _obs246(cfg, full) -> None:
  """5-step action history: observation 126 -> 246."""
  cfg.observations["actor"].terms["actions"].history_length = 5


def _knee(cfg, full) -> None:
  """Knee effort limit 70 N.m instead of 100, through the action scale."""
  for j in ("L_KNEE_P", "R_KNEE_P"):
    cfg.actions["joint_pos"].scale[j] = 0.00525


def _mirror(cfg, full) -> None:
  """Handled in rl_cfg via mirror_enabled(); nothing to do on the env."""


def _rand(cfg, full) -> None:
  """The randomisation policy 0 did not have, at the softened values."""
  for name in EXTRA_EVENTS:
    if name in full["events"]:
      cfg.events[name] = full["events"][name]
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)


def _feet(cfg, full) -> None:
  """The foot-reward recalibrations, as a block: they were tuned together."""
  r = cfg.rewards
  r["flat_support"].weight = -11.0
  r["flat_support"].params["command_name"] = "twist"
  r["flat_support"].params["corner_tolerance"] = 0.001
  r["foot_clearance"].weight = -10.0
  r["foot_slip"].weight = -0.5
  r["impact_vel"].weight = -2.0
  r["impact_vel"].params["limit"] = 0.15
  r["standing_single_support"].weight = -6.0
  r["air_time"].params["power"] = 1.0
  r["air_time"].params["touchdown_cost"] = 0.0
  r.pop("min_foot_height", None)
  r.pop("flat_touchdown", None)


def _prox(cfg, full) -> None:
  """The six proximity terms, asked for as hardware protection."""
  for name in PROXIMITY_TERMS:
    if name in full["rewards"]:
      cfg.rewards[name] = full["rewards"][name]


def _angmom(cfg, full) -> None:
  """angular_momentum -0.3 and ankle_roll_torque dropped."""
  cfg.rewards["angular_momentum"].weight = -0.3
  cfg.rewards.pop("ankle_roll_torque", None)


DELTAS = {
  "obs246": _obs246,
  "knee": _knee,
  "mirror": _mirror,
  "rand": _rand,
  "feet": _feet,
  "prox": _prox,
  "angmom": _angmom,
}


def apply_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Revert to policy 0, then re-apply the requested deviations in order."""
  steps = selection()
  if steps is None:
    return
  # Snapshot first: the rungs that re-add something restore it from here rather
  # than redeclaring it, so a rung cannot drift from what the config really says.
  full = {"rewards": copy.deepcopy(cfg.rewards), "events": copy.deepcopy(cfg.events)}
  _revert_to_policy0(cfg)
  for step in steps[1:]:
    DELTAS[step](cfg, full)
