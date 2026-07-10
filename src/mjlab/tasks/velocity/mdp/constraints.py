from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def cstr_joint_torque(
  env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cstr = torch.abs(asset.data.actuator_force[:, asset_cfg.joint_ids]) - limit
  return cstr


def cstr_joint_position(
  env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  # print(asset.data.joint_pos[:3, asset_cfg.joint_ids])
  # print(limit)
  cstr = limit - asset.data.joint_pos[:, asset_cfg.joint_ids]
  return cstr


def cstr_impact_vel(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  limit: float,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]

  violation = torch.zeros_like(first_contact, dtype=torch.float)  # [B, N]
  sides = ["left", "right"]
  tips = ["toes", "heel"]
  # print("--------")
  for i, side in enumerate(sides):
    for j, tip in enumerate(tips):
      footVelSensor: Entity = env.scene[f"robot/{side}_foot_{tip}_lin_vel"]
      footVelData = footVelSensor.data
      assert footVelData is not None
      footVelNorm = torch.norm(footVelData, dim=1)
      landingVel = footVelNorm * first_contact[:, i * 2 + j].float()  # [B]
      violation[:, i * 2 + j] = landingVel - limit
      # print(f"robot/{side}_foot_{tip}_lin_vel", landingVel)

  num_landings = torch.sum(first_contact.float())
  mean_landing_vel = torch.sum(violation + limit) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_vel_mean"] = mean_landing_vel

  # if torch.any(mean_landing_vel > 0):
  #  print(mean_landing_vel)

  return violation


def cstr_gait(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = contact_sensor.data.found > 0  # [B, N]

  from mjlab.tasks.velocity.mdp.velocity_command import PhaseCommand  # noqa: PLC0415
  phase_term: CommandTerm | None = env.command_manager.get_term("phase")
  assert type(phase_term) is PhaseCommand
  phase_episode = phase_term.phase_episode

  contactPhases = (
    2
    * torch.pi
    * (phase_episode.unsqueeze(1) + torch.tensor([0.0, 0.5], device=env.device))
  )
  feetCtcProba = env.getContactProbability(contactPhases)

  swingViolation = in_contact * (feetCtcProba < 0.1)
  stanceViolation = (~in_contact) * (feetCtcProba > 0.9)
  violation = torch.sum(torch.logical_or(swingViolation, stanceViolation), dim=1) / 2

  # print(in_contact)
  # print(feetCtcProba)
  # print("====")

  return violation


def cstr_feet_distance(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]

  # For simplicity we consider only two first feet
  foot_diff_xy = foot_pos_xy[:, 0] - foot_pos_xy[:, 1]  # [B, 2]
  print(foot_diff_xy)

  distance = torch.norm(foot_diff_xy, dim=-1)  # [B]
  violation = distance - limit
  return violation


def cstr_joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  violation = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  violation += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)

  return violation


def cstr_ankle_deviation(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Constrain ankle pitch/roll deviation from default pose.

  This helps avoid edge-walking postures where the robot loads the side of the foot.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  if default_joint_pos is None:
    raise RuntimeError("default_joint_pos is required for ankle deviation constraint.")
  deviation = torch.abs(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.clamp(deviation - limit, min=0.0)
