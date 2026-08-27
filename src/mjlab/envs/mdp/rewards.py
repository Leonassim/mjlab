"""Useful methods for MDP rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive."""
  return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize terminated episodes that don't correspond to episodic timeouts."""
  return env.termination_manager.terminated.float()


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint torques applied on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1
  )


def joint_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint velocities on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2_standing(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """joint_vel_l2, charged only while the robot is asked to stand still.

  Arm swing is not decoration. A leg swinging forward torques the body about the
  vertical axis and a natural gait cancels it with the arms in counter-phase;
  forbid that and the hip yaw absorbs all of it. Measured on RHPS1 with the
  upper body penalised at every instant: CROTCH_Y ran at 0.78 of its torque
  limit against the knee's 0.57, error_vel_yaw sat at 0.29, and the stride
  would not grow past 7 cm however hard it was paid for. The arms clipped too --
  two thirds of all clipping -- because holding still against that momentum is
  itself a static strain.

  Ungated, this term buys a quiet robot at the price of a gait that cannot
  lengthen its step. Gated, it keeps the thing actually wanted -- no fidgeting
  at rest -- and returns the mechanism the walk needs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  cost = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
  command = env.command_manager.get_command(command_name)
  if command is None:
    return cost
  moving = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return cost * (moving <= command_threshold).float()


def joint_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint accelerations on the articulation using L2 squared kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the acceleration of the actions using L2 squared kernel.

  Operates on raw policy output (before per-term scale/offset).
  """
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc), dim=1)


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


class posture:
  """Penalize the deviation of the joint positions from the default positions.

  Note: This is implemented as a class so that we can resolve the standard deviation
  dictionary into a tensor and thereafter use it in the __call__ method.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused.
    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))




class joint_effort_l2:
  """Penalize actuator force for actuators matching a name pattern."""

  def __init__(self, cfg: "RewardTermCfg", env: ManagerBasedRlEnv):
    import re

    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    pattern = cfg.params.get("actuator_pattern", r".*")
    regex = re.compile(pattern)
    self._indices = [
      i for i, name in enumerate(asset.actuator_names) if regex.search(name)
    ]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    actuator_pattern: str = r".*",
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.actuator_force[:, self._indices]), dim=1)


def joint_torque_limit_margin_penalty(
  env: ManagerBasedRlEnv,
  soft_ratio: float = 0.7,
  power: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize actuator forces as they approach their torque limits."""
  asset: Entity = env.scene[asset_cfg.name]
  actuator_names = asset.actuator_names
  active_local_ids = [
    idx for idx, name in enumerate(actuator_names) if not name.endswith("_motor")
  ]
  if not active_local_ids:
    active_local_ids = list(range(len(actuator_names)))

  active_local_ids_t = torch.tensor(
    active_local_ids, device=env.device, dtype=torch.long
  )
  actuator_force = torch.abs(asset.data.actuator_force[:, active_local_ids_t])
  ctrl_ids = asset.indexing.ctrl_ids[active_local_ids_t]
  force_limits = env.sim.model.actuator_forcerange[:, ctrl_ids, 1]

  eps = 1e-6
  valid = force_limits > eps
  normalized = torch.zeros_like(actuator_force)
  normalized[valid] = actuator_force[valid] / force_limits[valid]

  denom = max(1.0 - soft_ratio, eps)
  excess = torch.clamp((normalized - soft_ratio) / denom, min=0.0)
  if power != 1.0:
    excess = torch.pow(excess, power)

  env.extras["log"]["Metrics/torque_limit_ratio_mean"] = torch.mean(normalized)
  env.extras["log"]["Metrics/torque_limit_ratio_max"] = torch.max(normalized)
  # Per actuator: the aggregate hides that the load is concentrated. A raw cost
  # of 3.06 against a mean ratio of 0.36 means a handful of joints ride the
  # limit while the rest cruise, and which ones decides whether a global torque
  # penalty is even the right tool.
  per_joint = torch.mean(normalized, dim=0)
  for k, local_id in enumerate(active_local_ids):
    env.extras["log"][f"TorqueRatio/{actuator_names[local_id]}"] = per_joint[k]

  # actuator_force is what MuJoCo applied, already clamped to forcerange, so
  # normalized cannot exceed 1 and the max reads 1.0000 on every line whether
  # one sample clips or half of them do. The level of torque is not the problem
  # -- a longer step costs more, and riding the limit is fine. Demanding past it
  # is not: the action is silently truncated and the policy is controlling a
  # plant it cannot feel. This is that fraction.
  saturated = (normalized >= 0.99).float()
  env.extras["log"]["Metrics/torque_saturated_frac"] = torch.mean(saturated)
  per_joint_sat = torch.mean(saturated, dim=0)
  leg_idx = []
  for k, local_id in enumerate(active_local_ids):
    env.extras["log"][f"TorqueSat/{actuator_names[local_id]}"] = per_joint_sat[k]
    if any(t in actuator_names[local_id] for t in ("CROTCH", "KNEE", "ANKLE")):
      leg_idx.append(k)
  # Legs alone. The aggregate above is two thirds upper body -- wrists, elbows
  # and shoulders saturate their own small actuators -- and torque is NOT shared
  # between joints, so a clipped wrist takes nothing from the knee. Guarding the
  # aggregate therefore reads mostly a quantity that cannot constrain the
  # stride, and hides the one that can.
  if leg_idx:
    env.extras["log"]["Metrics/torque_saturated_frac_legs"] = torch.mean(
      per_joint_sat[torch.tensor(leg_idx, device=per_joint_sat.device)]
    )
  # Upper body, separately. Splitting the legs out of the aggregate was right --
  # a clipped wrist takes nothing from the knee -- but it left the upper body
  # with no criterion at all, and Leo names its torque as something to watch.
  # Two thirds of all clipping lives here; unwatched, an arm can ride its limit
  # for a whole campaign without a single number moving.
  upper_idx = [k for k in range(len(active_local_ids)) if k not in set(leg_idx)]
  if upper_idx:
    env.extras["log"]["Metrics/torque_saturated_frac_upper"] = torch.mean(
      per_joint_sat[torch.tensor(upper_idx, device=per_joint_sat.device)]
    )
  return torch.sum(excess, dim=1)




class joint_torque_rate_l2:
  """Penalize step-to-step changes in actuator torque."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]

    joint_names = asset_cfg.joint_names if asset_cfg.joint_names else (".*",)
    actuator_ids, _ = asset.find_actuators(joint_names)
    if not actuator_ids:
      joint_ids, _ = asset.find_joints(joint_names)
      actuator_ids = joint_ids
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)
    self._prev_tau = torch.zeros(
      (env.num_envs, len(actuator_ids)), device=env.device, dtype=torch.float
    )
    self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_tau[env_ids] = 0.0
    self._initialized[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]
    delta = tau - self._prev_tau
    reward = torch.sum(torch.square(delta), dim=1)
    first = ~self._initialized
    reward = torch.where(first, torch.zeros_like(reward), reward)
    self._prev_tau[:] = tau
    self._initialized[:] = True
    return reward


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
