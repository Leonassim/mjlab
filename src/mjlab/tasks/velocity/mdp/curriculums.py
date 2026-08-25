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


def torque_feasibility_progress(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start_step: int,
  end_step: int,
  start_progress: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Tighten the torque-feasibility projection from loose to full.

  The actuator's effective cap is ``torque_feasibility_ratio / progress``, so
  ``start_progress=0.25`` begins at 4x the configured ratio and ``1.0`` reaches
  it exactly.

  Registering this is *optional* and deliberately not wired by default. The
  projection is a constraint on the action space, not a penalty routed through
  the advantage, so it cannot produce the timidity loop that forced every
  pd_demand_excess ramp -- and the whole premise is that exploration should
  never leave the feasible set, which a ramp partially defeats. Wire it only if
  a first run at full projection fails to form a gait at all, i.e. if the
  projection turns out to cap swing velocity before the policy has found a
  stride worth swinging.
  """
  del env_ids  # Unused -- progress is global, not per-env.
  step = env.common_step_counter
  frac = min(1.0, max(0.0, (step - start_step) / max(end_step - start_step, 1)))
  progress = float(start_progress + (1.0 - start_progress) * frac)
  robot: Entity = env.scene[asset_cfg.name]
  for act in robot.actuators:
    if isinstance(act, FiniteDifferencePdActuator):
      act.torque_feasibility_progress = progress
  return torch.tensor([progress])


class PushStage(TypedDict):
  step: int
  scale: float




class StandingEnvsStage(TypedDict):
  step: int
  value: float




def step_target_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  stages: list[dict],
) -> torch.Tensor:
  """Walk com_step_progress's targets up in stages.

  A fixed target stops pulling the moment it is reached -- the clamp is there
  precisely so it cannot drag the policy past feasibility, which means the
  target has to move if the stride is to keep growing. Measured 3.9 cm against
  a 5 cm target: ratio 0.78, the pull was about to die.

  Same mutate-in-place mechanism as air_time_target_curriculum, and the same
  discipline: each stage only fires after the gait has had a window to
  consolidate at the previous one, so the current behaviour never becomes
  suddenly unprofitable.
  """
  del env_ids
  cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      cfg.params["target_distance"] = stage["target_distance"]
      cfg.params["target_period"] = stage["target_period"]
  return torch.tensor([cfg.params["target_distance"]])


_RELATIVE_BASE: dict[tuple[str, str], tuple[int, int]] = {}


def reward_param_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param: str,
  stages: list[dict],
  relative: bool = False,
) -> torch.Tensor:
  """Walk one reward parameter through stages. Same mechanism as the others.

  Written for foot height, where the target sat at 30 mm against a measured
  3.1 mm peak. A target that far out is not a goal, it is a constant: the
  policy pays the penalty and the gradient tells it nothing about which way is
  cheaper. The step-length ladder only worked because its first target was near
  what the gait already did.
  """
  del env_ids
  cfg = env.reward_manager.get_term_cfg(reward_name)
  # common_step_counter carries over when a run is resumed, so an absolute
  # ladder lands on its last stage immediately -- which is exactly the failure
  # this ladder exists to avoid, since stage 0 is deliberately set near what the
  # gait already does. Count from the first call instead.
  # `step` counts CALLS to this function -- one per iteration -- not
  # common_step_counter. Two attempts to offset that counter failed for the
  # same reason: it is restored from the checkpoint *after* the curriculum's
  # first call, so a base captured then reads 0 while the counter itself jumps
  # to 417820, and every stage fires at once. A call counter owes nothing to
  # checkpoint state, and stages read as iterations, which is how they are
  # reasoned about anyway.
  if relative:
    # Two failed attempts before this one. common_step_counter is restored from
    # the checkpoint *after* the curriculum's first call, so a base captured
    # there reads 0 against a counter already at 417820 and every stage fires
    # at once. Counting calls instead was worse: the manager calls this ~25x
    # per iteration, so the ladder crossed three stages in 40.
    #
    # Use the counter, and re-baseline when it jumps by more than a step's
    # worth -- which is exactly and only the restore.
    key = (reward_name, param)
    counter = env.common_step_counter
    base, last = _RELATIVE_BASE.get(key, (counter, counter))
    if counter - last > 1000:
      base = counter
    _RELATIVE_BASE[key] = (base, counter)
    calls = counter - base
  else:
    calls = env.common_step_counter
  for stage in stages:
    if calls > stage["step"]:
      cfg.params[param] = stage["value"]
  return torch.tensor([float(cfg.params[param])])

_GATE_STATE: dict[str, dict] = {}


def metric_gated_param_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param: str,
  metric: str,
  values: list[float],
  hold: int = 20,
  lower_is_better: bool = True,
) -> torch.Tensor:
  """Tighten one reward parameter, one notch at a time, on measured success.

  Every other curriculum here advances on a step counter, which is the flaw
  this project keeps paying for: stages march past while the quantity they ask
  for does not move, and the late ones become constants.

  It also fixes the failure mode of a step change. Asking for a softer landing
  by tripling the penalty, or by moving the threshold 0.20 -> 0.16 in one go,
  was answered three times running by the policy simply not landing -- the
  escape became cheaper than the demand the instant the demand appeared. One
  notch of 0.01, taken only once the gait already meets the current notch, is
  never a big enough jump for the escape to pay.

  Advances when the metric has held on the good side of the CURRENT value for
  `hold` consecutive calls. It never goes back: a notch regained is not the
  same evidence as a notch held.
  """
  del env_ids
  cfg = env.reward_manager.get_term_cfg(reward_name)
  key = f"{reward_name}.{param}"
  st = _GATE_STATE.setdefault(key, {"i": 0, "streak": 0})
  cfg.params[param] = values[st["i"]]

  v = env.extras.get("log", {}).get(metric)
  if v is not None:
    v = float(v)
    good = (v <= values[st["i"]]) if lower_is_better else (v >= values[st["i"]])
    st["streak"] = st["streak"] + 1 if good else 0
    if st["streak"] >= hold and st["i"] + 1 < len(values):
      st["i"] += 1
      st["streak"] = 0
      cfg.params[param] = values[st["i"]]
  return torch.tensor([float(values[st["i"]])])
