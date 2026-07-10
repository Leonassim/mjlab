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


class electrical_power_cost:
  """Penalize electrical power consumption of actuators."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, _ = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, self._joint_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    mech = tau * qd
    mech_pos = torch.clamp(mech, min=0.0)  # Don't penalize regen.
    return torch.sum(mech_pos, dim=1)


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
  return torch.sum(excess, dim=1)


def joints_action_acc_l2(
  env: ManagerBasedRlEnv,
  joint_indices: list[int],
) -> torch.Tensor:
  """Penalize action acceleration for a specific subset of joints."""
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc[:, joint_indices]), dim=1)


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
