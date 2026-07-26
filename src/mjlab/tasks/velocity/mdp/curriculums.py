from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.actuator.finite_difference_pd_actuator import FiniteDifferencePdActuator
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def air_time_target_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  stages: list[dict],
  gait_reward_name: str | None = None,
) -> torch.Tensor:
  """Raise the air-time bonus ceiling (threshold_max) in stages, and (if
  ``gait_reward_name`` is given) keep the gait_phase clock's period in
  lockstep with it, so the two never drift into the same conflict this was
  written to fix.

  Unlike a naive threshold ramp starting at step 0 (tried and reverted: the
  early high-exploration phase locked onto whatever short steps saturated a
  low initial ceiling before later stages could pull toward longer strides),
  each stage here only fires once the gait has had a window to consolidate
  at the previous ceiling, and pairs threshold_max with touchdown_cost so
  the profitability break-even (threshold_max * sqrt(touchdown_cost)) stays
  near the trailing operating point instead of jumping past it -- raising
  the ceiling without ever making current behavior suddenly unprofitable.
  overflow_threshold moves with it to keep the anti-hover guard's margin
  proportional. Same mutate-in-place mechanism as ``reward_weight``.

  ``gait_reward_name``: retimes gait_phase's clock period in lockstep with
  threshold_max at every stage (period = threshold_max / swing_ratio),
  fixing the class of bug where the two drift out of sync (observed
  2026-07-24 between air_time and stride_frequency_target's independently-
  fixed period, a ~25% stance / 75% swing hop-like gait). Tried and
  reverted 2026-07-25, though: retiming the clock's cadence mid-training --
  even a single, modest change -- turned out to invalidate whatever the
  actor had learned to do with a given phase observation (the sin/cos
  value it read no longer means the same point in the cycle), which is a
  qualitatively different kind of disruption than a reward-weight change.
  Run 2026-07-25_18-33-25 was stable through the equivalent point with a
  *fixed* f_gait, then entered a slow, never-recovering fell_down climb
  starting ~600 iterations after the first coupled retiming, compounding
  into full collapse ~1700 iterations later. Left here (unused by the
  current env_cfgs.py registration) in case a *gradual* retime (ramped
  over many steps rather than snapped in one) is worth trying later --
  don't wire this back up as an instant jump.
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.params["threshold_max"] = stage["threshold_max"]
      reward_term_cfg.params["touchdown_cost"] = stage["touchdown_cost"]
      reward_term_cfg.params["overflow_threshold"] = stage["overflow_threshold"]
  if gait_reward_name is not None:
    gait_term_cfg = env.reward_manager.get_term_cfg(gait_reward_name)
    swing_ratio = gait_term_cfg.params["swing_ratio"]
    period = reward_term_cfg.params["threshold_max"] / swing_ratio
    gait_term_cfg.params["f_gait"] = 1.0 / period
  return torch.tensor([reward_term_cfg.params["threshold_max"]])


def min_foot_height_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  stages: list[dict],
) -> torch.Tensor:
  """Raise both the min_foot_height weight and its min_height target together.

  Fixing min_height at the final target (0.08m) from the start and only
  ramping the weight (tried first) plateaued Metrics/peak_height_mean at
  ~0.006-0.009m for 5000+ iterations regardless of weight: the target was
  simply out of reach the whole time, so a bigger weight just meant a bigger
  constant penalty, not a closer target to actually climb toward. Same
  break-even idea as ``air_time_target_curriculum``: each stage's target
  should be within reach of the previous stage's converged behavior, not
  fixed far ahead of it from step 0.
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
      reward_term_cfg.params["min_height"] = stage["min_height"]
  return torch.tensor([reward_term_cfg.params["min_height"]])


def velocity_damper_progress(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start_step: int,
  end_step: int,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Linearly ramp the velocity damper from 0 → 1 between ``start_step`` and ``end_step``.

  Sets ``velocity_damper_progress`` on every ``FiniteDifferencePdActuator`` of
  the robot entity so the safety projection gradually tightens to match the
  mc_rtc QP KinematicsConstraint (``di=0.4``, ``ds=0.01``, ``vel=0.9``).
  At ``start_step`` the damper is inactive; at ``end_step`` it is fully active.
  """
  del env_ids  # Unused — progress is global, not per-env.
  step = env.common_step_counter
  progress = float(
    min(1.0, max(0.0, (step - start_step) / max(end_step - start_step, 1)))
  )
  robot: Entity = env.scene[asset_cfg.name]
  for act in robot.actuators:
    if isinstance(act, FiniteDifferencePdActuator):
      act.velocity_damper_progress = progress
  return torch.tensor([progress])


class PushStage(TypedDict):
  step: int
  scale: float


def push_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  max_velocity_range: dict[str, tuple[float, float]],
  stages: list[PushStage],
) -> torch.Tensor:
  """Scale push perturbation velocity range based on training progress."""
  del env_ids
  scale = 0.0
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      scale = stage["scale"]
  event_cfg = env.event_manager.get_term_cfg(event_name)
  event_cfg.params["velocity_range"] = {
    k: (v[0] * scale, v[1] * scale) for k, v in max_velocity_range.items()
  }
  return torch.tensor([scale])


class StandingEnvsStage(TypedDict):
  step: int
  value: float


def standing_envs_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[StandingEnvsStage],
) -> torch.Tensor:
  """Decrease the proportion of standing (zero-command) envs over training."""
  del env_ids
  value = stages[0]["value"]
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      value = stage["value"]
  command_term = env.command_manager.get_term(command_name)
  command_term.cfg.rel_standing_envs = value
  return torch.tensor([value])
