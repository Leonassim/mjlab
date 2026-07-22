from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def leg_proximity_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_dist: float = 0.01,
) -> torch.Tensor:
  """Penalize leg-leg clearance below ``min_dist``.

  Mirrors the deployment QP's self-collision damper margin so the policy
  keeps out of its activation zone. Requires a contact sensor with a ``dist``
  field on geoms whose ``gap`` extends detection beyond ``min_dist``
  (forceless proximity contacts). Penalty ramps linearly from 0 at
  ``min_dist`` to 1 at touch, and keeps growing with penetration.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  dist = sensor.data.dist
  if found is None or dist is None:
    raise RuntimeError("leg_proximity_cost needs 'found' and 'dist' fields.")
  violation = torch.clamp(min_dist - dist, min=0.0) * (found > 0).float()
  return torch.sum(violation, dim=1) / min_dist


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))

def _split_foot_contact_tensors(
  sensor: ContactSensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  found = sensor.data.found
  if found is None:
    raise RuntimeError("Contact sensor must provide 'found'.")
  if found.shape[1] < 8:
    raise RuntimeError("Split-foot contact rewards expect 8 contact slots.")
  split_found = found[:, :8].view(found.shape[0], 2, 4)
  contact_count = torch.sum((split_found > 0).float(), dim=2)
  foot_in_contact = (contact_count > 0).float()
  return contact_count, foot_in_contact


def split_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  overflow_threshold: float | None = None,
  overflow_weight_ratio: float = 1.0,
  command_name: str | None = None,
  command_threshold: float = 0.5,
  power: float = 1.0,
  touchdown_cost: float = 0.0,
) -> torch.Tensor:
  """Reward per-foot air time aggregated from 4 split contacts per foot.

  Event-based: pays ``(min(last_air_time, threshold_max) /
  threshold_max) ** power - touchdown_cost`` once, at touchdown (contrast
  with the dense, potential-based ``split_feet_air_time_dense`` below, which
  pays the same total but spread across every airborne step). Landings
  whose air time is below ``threshold_min`` earn nothing.

  ``overflow_threshold`` sets the per-foot air-time limit beyond which a
  penalty fires each step (deters hover exploits); ``no_double_flight``
  handles the both-feet-airborne exploit independently.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  last_air_time = sensor.data.last_air_time
  if current_air_time is None or last_air_time is None:
    raise RuntimeError("Contact sensor must have track_air_time=True.")
  if current_air_time.shape[1] < 8:
    raise RuntimeError("Split-foot air-time reward expects 8 contact slots.")

  split_air = current_air_time[:, :8].view(current_air_time.shape[0], 2, 4)
  _, foot_in_contact = _split_foot_contact_tensors(sensor)
  foot_in_air = 1.0 - foot_in_contact
  foot_air_time = torch.max(split_air, dim=2).values * foot_in_air

  # Air time of the stride that just ended. current_air_time is zeroed at the
  # contact step, so read last_air_time from the slots that landed within the
  # last step.
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  split_first = first_contact[:, :8].view(first_contact.shape[0], 2, 4).float()
  split_last_air = last_air_time[:, :8].view(last_air_time.shape[0], 2, 4)
  foot_last_air = torch.max(split_last_air * split_first, dim=2).values
  foot_landed = (foot_last_air > threshold_min).float()
  value = (torch.clamp(foot_last_air, max=threshold_max) / threshold_max) ** power
  landing_reward = torch.sum((value - touchdown_cost) * foot_landed, dim=1)
  ot = overflow_threshold if overflow_threshold is not None else 2.0 * threshold_max
  overflow = torch.clamp(foot_air_time - ot, min=0.0) * foot_in_air
  overflow_penalty = torch.sum(overflow, dim=1)
  num_in_air = torch.sum(foot_in_air)
  mean_air_time = torch.sum(foot_air_time) / torch.clamp(num_in_air, min=1.0)
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

  # Only the landing bonus is command-gated: the overflow penalty must also
  # apply to standing envs, otherwise hovering on one foot at zero command is
  # free.
  reward = landing_reward
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      reward = reward * (total_command > command_threshold).float()
  return reward - overflow_weight_ratio * overflow_penalty


def split_feet_air_time_dense(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  overflow_threshold: float | None = None,
  overflow_weight_ratio: float = 1.0,
  command_name: str | None = None,
  command_threshold: float = 0.5,
  power: float = 1.0,
  touchdown_cost: float = 0.0,
) -> torch.Tensor:
  """Potential-based dense air-time shaping (Ng, Harada & Russell 1999).

  Pays dPhi/dt every step a foot is airborne, where Phi(a) = (min(a,
  threshold_max)/threshold_max)**power. By the telescoping-sum identity this
  sums to Phi(T) over a complete swing of duration T -- policy-invariant,
  just paid continuously instead of once at touchdown.

  Ng et al.'s proof requires Phi(terminal state) = 0, or an episode ending
  mid-swing banks credit for a landing that never happened. On termination
  (not timeout), the potential for any still-airborne foot is clawed back to
  restore that invariant.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  last_air_time = sensor.data.last_air_time
  if current_air_time is None or last_air_time is None:
    raise RuntimeError("Contact sensor must have track_air_time=True.")
  if current_air_time.shape[1] < 8:
    raise RuntimeError("Split-foot air-time reward expects 8 contact slots.")

  split_air = current_air_time[:, :8].view(current_air_time.shape[0], 2, 4)
  _, foot_in_contact = _split_foot_contact_tensors(sensor)
  foot_in_air = 1.0 - foot_in_contact
  foot_air_time = torch.max(split_air, dim=2).values * foot_in_air  # [B, 2]

  clamped_a = torch.clamp(foot_air_time, max=threshold_max)
  d_phi_da = power * torch.pow(clamped_a / threshold_max, power - 1.0) / threshold_max
  d_phi_da = torch.where(
    foot_air_time < threshold_max, d_phi_da, torch.zeros_like(d_phi_da)
  )
  dense_bonus = torch.sum(d_phi_da * env.step_dt * foot_in_air, dim=1)

  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  split_first = first_contact[:, :8].view(first_contact.shape[0], 2, 4).float()
  split_last_air = last_air_time[:, :8].view(last_air_time.shape[0], 2, 4)
  foot_last_air = torch.max(split_last_air * split_first, dim=2).values
  foot_landed = (foot_last_air > threshold_min).float()
  touchdown_fee = torch.sum(touchdown_cost * foot_landed, dim=1)

  ot = overflow_threshold if overflow_threshold is not None else 2.0 * threshold_max
  overflow = torch.clamp(foot_air_time - ot, min=0.0) * foot_in_air
  overflow_penalty = torch.sum(overflow, dim=1)

  num_in_air = torch.sum(foot_in_air)
  mean_air_time = torch.sum(foot_air_time) / torch.clamp(num_in_air, min=1.0)
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

  reward = dense_bonus - touchdown_fee
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      reward = reward * (total_command > command_threshold).float()
  reward = reward - overflow_weight_ratio * overflow_penalty

  terminated = env.termination_manager.terminated
  if terminated is not None:
    phi_value = torch.pow(clamped_a / threshold_max, power)  # Phi(current_air_time)
    clawback = torch.sum(phi_value * foot_in_air, dim=1) * terminated.float()
    reward = reward - clawback
  return reward


def feet_air_time_symmetry(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Penalize left/right imbalance of the last completed swing durations.

  Nothing else in the reward compares the two feet, so a gait where one leg
  collects long-stride bonuses while the other makes minimal support hops is
  otherwise profitable. Cost = |last_air_L - last_air_R| every step (per-foot
  air time of the last completed swing, max over the 4 split slots).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  last_air = sensor.data.last_air_time
  if last_air is None:
    raise RuntimeError("Contact sensor must have track_air_time=True.")
  if last_air.shape[1] < 8:
    raise RuntimeError("Split-foot symmetry reward expects 8 contact slots.")
  split = last_air[:, :8].view(last_air.shape[0], 2, 4)
  foot_last = torch.max(split, dim=2).values  # [B, 2]
  cost = torch.abs(foot_last[:, 0] - foot_last[:, 1])
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      cost = cost * (total_command > command_threshold).float()
  return cost


def no_double_flight_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize phases where both feet are simultaneously off the ground."""
  sensor: ContactSensor = env.scene[sensor_name]
  _, foot_in_contact = _split_foot_contact_tensors(sensor)  # [B, 2]
  no_contact = (torch.sum(foot_in_contact, dim=1) == 0).float()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      no_contact = no_contact * (total_command > command_threshold).float()

  env.extras["log"]["Metrics/double_flight_rate"] = torch.mean(no_contact)
  return no_contact


def standing_single_support_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize standing on a single foot when the commanded motion is near zero."""
  sensor: ContactSensor = env.scene[sensor_name]
  _, foot_in_contact = _split_foot_contact_tensors(sensor)  # [B, 2]
  num_feet_in_contact = torch.sum(foot_in_contact, dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  standing = (total_command <= command_threshold).float()

  one_foot = (num_feet_in_contact == 1).float()
  no_feet = (num_feet_in_contact == 0).float()
  cost = (one_foot + 4.0 * no_feet) * standing
  env.extras["log"]["Metrics/standing_single_support_rate"] = torch.mean(
    (one_foot + no_feet) * standing
  )
  return cost


class pd_demand_excess:
  """Penalize the smoothed unclamped PD torque demand beyond the effort limit.

  The training actuator clamps kp*(q*-q)+kd*(v*-v) at effort_limit, so the
  policy silently learns to lean on saturation (torque_limit_ratio_max
  pinned at 1.0) -- and the applied-torque margin penalty cannot see past
  ratio 1. Deployment then diverges wherever the sim/hardware does not clamp
  identically (observed: mc_mujoco blow-up with 287 N.m demanded on a 35 N.m
  hip yaw).

  The demand is smoothed with an EMA (~time constant ``ema_dt`` seconds,
  ~40 ms) before the ratio test: a single noisy step's demand largely
  averages out over that window, so an elevated EMA reflects a state+action
  pair the policy actually committed to for several consecutive steps, not
  one-off exploration jitter -- whatever produced it (nominal gait, a fall,
  a recovery attempt), state + chosen action + dt deterministically imply
  the PD demand, and it must fit under the real limit unconditionally in
  the deployed policy, since real hardware won't clip it the way the
  training actuator does.

  In the env config this term's weight is ramped in late by curriculum, not
  active from step 0: the training actuator's clamp means the clipped
  execution is always torque-feasible, so a feasible action already exists
  for whatever dynamic the policy discovers with free exploration -- the
  late ramp squeezes the policy into producing that action directly once a
  gait exists, instead of constraining the exploration that found it.

  Per joint: excess = clamp(|EMA(demand)| / limit - soft_ratio, 0, cap).
  Logs Metrics/pd_demand_ratio_mean/max for hardware-readiness tracking.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._ema: torch.Tensor | None = None
    self._env = env

  def reset(self, env_ids: torch.Tensor) -> None:
    if self._ema is not None:
      self._ema[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float = 1.0,
    cap: float = 1.0,
    ema_dt: float = 0.04,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    data = asset.data
    demands = []
    limits = []
    for act in asset._actuators:
      stiffness = getattr(act, "stiffness", None)
      if stiffness is None:
        continue
      ids = act.target_ids
      q_err = data.joint_pos_target[:, ids] - data.joint_pos[:, ids]
      vel_target = getattr(act, "_desired_velocity_target", None)
      if vel_target is None:
        vel_target = torch.zeros_like(q_err)
      v_err = vel_target - data.joint_vel[:, ids]
      demands.append(stiffness * q_err + act.damping * v_err)
      limits.append(act.force_limit)
    demand = torch.cat(demands, dim=1)
    limit = torch.cat(limits, dim=1)

    if self._ema is None or self._ema.shape != demand.shape:
      self._ema = torch.zeros_like(demand)
    alpha = float(env.step_dt) / max(ema_dt, float(env.step_dt))
    self._ema += alpha * (demand - self._ema)

    ratio = torch.abs(self._ema) / torch.clamp(limit, min=1e-6)
    excess = torch.clamp(ratio - soft_ratio, min=0.0, max=cap)
    env.extras["log"]["Metrics/pd_demand_ratio_max"] = ratio.max()
    env.extras["log"]["Metrics/pd_demand_ratio_mean"] = ratio.mean()
    return torch.sum(excess, dim=1)


def standing_joint_vel_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint velocities when the commanded motion is near zero.

  At zero command the robot should hold still; this taxes the residual
  oscillation directly in joint space without touching the walking gait.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel_sq = torch.sum(
    torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
  )
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (total_command <= command_threshold).float()
  return joint_vel_sq * standing


def foot_flat_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize sole tilt of the feet relative to the world horizontal.

  The foot links' frames are world-aligned when the sole is flat (the leg
  pitch chain sums to zero in the keyframe), so the XY components of gravity
  projected into the foot frame measure sin(tilt). Applied in every phase,
  unlike the contact-count-based flat_support_penalty which only fires at
  touchdown.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
  batch, num_feet = body_quat_w.shape[0], body_quat_w.shape[1]
  gravity_w = asset.data.gravity_vec_w  # [B, 3]
  gravity_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4),
    gravity_w[:, None, :].expand(batch, num_feet, 3).reshape(-1, 3),
  ).view(batch, num_feet, 3)
  gravity_b = gravity_b / torch.clamp(
    torch.norm(gravity_b, dim=-1, keepdim=True), min=1e-6
  )
  tilt = torch.norm(gravity_b[..., :2], dim=-1)  # [B, N], sin(tilt) per foot
  env.extras["log"]["Metrics/foot_tilt_mean"] = torch.mean(tilt)
  return torch.sum(tilt, dim=1)


def feet_clearance_velocity_weighted(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height (absolute z), weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_distance_penalty(
  env: ManagerBasedRlEnv,
  target_distance: float,
  max_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet being too far apart in the horizontal plane."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  # Expect exactly two sites: left_foot and right_foot.
  feet_distance = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # [B]
  too_wide = torch.relu(feet_distance - max_distance)
  # Keep a tiny preference around target_distance without dominating gait.
  around_target = 0.1 * torch.square(feet_distance - target_distance)
  env.extras["log"]["Metrics/feet_distance_mean"] = torch.mean(feet_distance)
  return torch.square(too_wide) + around_target


class split_feet_swing_height:
  """Split-contact version of swing-height reward aggregated per foot."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def getFootHeightWrtTerrain(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    asset: Entity = env.scene[asset_cfg.name]
    site_names = asset_cfg.site_names
    if site_names is None:
      raise RuntimeError("There is no site assigned to feet.")
    if isinstance(site_names, str):
      site_names = (site_names,)

    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    for i, name in enumerate(site_names):
      sensor = env.scene[f"{name}_scan"]
      assert isinstance(sensor, RayCastSensor)
      raycast_heights = sensor.data.hit_pos_w[..., 2]
      foot_heights[:, i] -= raycast_heights.mean(dim=-1)
    return foot_heights

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    found = contact_sensor.data.found
    if found is None or found.shape[1] < 8:
      raise RuntimeError("Split-foot swing-height reward expects 8 contact slots.")
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = self.getFootHeightWrtTerrain(env, asset_cfg)

    split_found = found[:, :8].view(found.shape[0], 2, 4)
    foot_in_air = torch.all(split_found == 0, dim=2)
    first_contact = torch.any(
      contact_sensor.compute_first_contact(dt=self.step_dt)[:, :8].view(
        found.shape[0], 2, 4
      ),
      dim=2,
    )

    self.peak_heights = torch.where(
      foot_in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


class split_feet_min_swing_height(split_feet_swing_height):
  """Charge a one-sided minimum-peak-height deficit once per landing.

  Fires ``clamp(1 - peak/min_height, 0)`` at touchdown: air time is free,
  only landing with a low swing peak costs. Reuses the terrain-relative peak
  tracker of ``split_feet_swing_height``.
  """

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    found = contact_sensor.data.found
    if found is None or found.shape[1] < 8:
      raise RuntimeError("Split-foot min-height reward expects 8 contact slots.")
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = self.getFootHeightWrtTerrain(env, asset_cfg)

    split_found = found[:, :8].view(found.shape[0], 2, 4)
    foot_in_air = torch.all(split_found == 0, dim=2)
    first_contact = torch.any(
      contact_sensor.compute_first_contact(dt=self.step_dt)[:, :8].view(
        found.shape[0], 2, 4
      ),
      dim=2,
    )

    self.peak_heights = torch.where(
      foot_in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    deficit = torch.clamp(1.0 - self.peak_heights / min_height, min=0.0)
    cost = torch.sum(deficit * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
      peak_heights_at_landing
    ) / torch.clamp(num_landings, min=1)
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  standing_scale: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  standing = 1.0 - active
  scale = 1.0 + standing_scale * standing
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * scale
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def split_feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  standing_scale: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Split-contact version of foot-slip reward aggregated per foot."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()

  _, foot_in_contact = _split_foot_contact_tensors(contact_sensor)  # [B, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  vel_xy_norm_sq = torch.square(vel_xy_norm)
  standing = 1.0 - active
  scale = 1.0 + standing_scale * standing
  cost = torch.sum(vel_xy_norm_sq * foot_in_contact, dim=1) * scale
  num_in_contact = torch.sum(foot_in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * foot_in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel

  return cost


def stance_action_acc_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  left_joint_indices: list[int],
  right_joint_indices: list[int],
) -> torch.Tensor:
  """Penalize action acceleration only for the joints of the stance (contact) leg.

  Swing-leg joints are excluded, allowing vigorous foot lifting without penalty
  while still preventing oscillations on the weight-bearing leg.

  The contact sensor must provide a ``found`` tensor with at least 8 columns:
  columns 0-3 for the left foot split patches, columns 4-7 for the right.
  """
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None or found.shape[1] < 8:
    return torch.zeros(env.num_envs, device=env.device)
  contacts = (found[:, :8] > 0).float()
  left_stance = (contacts[:, :4].sum(dim=1) > 0).float()
  right_stance = (contacts[:, 4:8].sum(dim=1) > 0).float()
  left_acc_sq = torch.sum(torch.square(action_acc[:, left_joint_indices]), dim=1)
  right_acc_sq = torch.sum(torch.square(action_acc[:, right_joint_indices]), dim=1)
  return left_stance * left_acc_sq + right_stance * right_acc_sq


def flat_support_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  required_contacts_per_foot: int = 4,
) -> torch.Tensor:
  """Enforce a strict 4-or-0 support rule for each foot.

  For each foot independently:
  - if no split contact zone is touching, cost is zero
  - if any split contact zone is touching, all four must be touching

  This is stricter than a generic edge-contact penalty and matches robots that
  must land and support weight with a flat sole rather than rolling over heel,
  toe, or edge.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    raise RuntimeError("Contact sensor must provide 'found' for flat support penalty.")
  if found.shape[1] < 8:
    raise RuntimeError("flat_support_penalty expects 8 split foot contacts.")

  contacts = (found[:, :8] > 0).float().view(found.shape[0], 2, 4)
  contact_count = torch.sum(contacts, dim=2)  # [B, 2]
  in_contact = (contact_count > 0).float()

  required_contacts = float(required_contacts_per_foot)
  deficit = torch.clamp(required_contacts - contact_count, min=0.0) / max(
    required_contacts, 1.0
  )
  cost = torch.sum(torch.square(deficit) * in_contact, dim=1)

  env.extras["log"]["Metrics/flat_support_contacts_mean"] = torch.sum(
    contact_count * in_contact
  ) / torch.clamp(torch.sum(in_contact), min=1.0)
  env.extras["log"]["Metrics/stance_contacts_mean"] = torch.sum(
    contact_count * in_contact
  ) / torch.clamp(torch.sum(in_contact), min=1.0)
  return cost


def impact_velocity(
  env: ManagerBasedRlEnv,
  limit: float,
  sensor_name: str,
  start_step: int = 0,
  pre_contact_limit: float | None = None,
  pre_contact_window_s: float = 0.0,
  always_limit: float | None = None,
  command_name: str | None = None,
  always_command_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalize foot linear velocity at landing, using last in-air velocity."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  found = contact_sensor.data.found
  if found is None:
    raise RuntimeError(
      "Contact sensor must provide 'found' to compute impact velocity."
    )

  eps = 1e-6
  use_pre_contact_window = pre_contact_limit is not None and pre_contact_window_s > 0.0
  window_steps = 0
  if use_pre_contact_window:
    window_steps = max(1, int(round(pre_contact_window_s / env.step_dt)))

  cost_per_slot = torch.zeros_like(first_contact, dtype=torch.float)
  landing_vel_per_slot = torch.zeros_like(first_contact, dtype=torch.float)
  pre_contact_cost = torch.zeros_like(first_contact, dtype=torch.float)
  pre_contact_peak_vel = torch.zeros_like(first_contact, dtype=torch.float)
  always_cost = torch.zeros_like(first_contact, dtype=torch.float)
  always_vel = torch.zeros_like(first_contact, dtype=torch.float)

  always_active: torch.Tensor | None = None
  if (
    always_limit is not None
    and command_name is not None
    and always_command_threshold > 0.0
  ):
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    always_active = (total_command >= always_command_threshold).float()

  # Assumed order of slots:
  # [left_foot1, left_foot2, left_foot3, left_foot4,
  #  right_foot1, right_foot2, right_foot3, right_foot4]
  slot_names = [
    "robot/left_foot_toes_lin_vel",
    "robot/left_foot_heel_lin_vel",
    "robot/left_foot_inner_lin_vel",
    "robot/left_foot_outer_lin_vel",
    "robot/right_foot_toes_lin_vel",
    "robot/right_foot_heel_lin_vel",
    "robot/right_foot_inner_lin_vel",
    "robot/right_foot_outer_lin_vel",
  ]

  for idx, sensor_path in enumerate(slot_names):
    vel_sensor: Entity = env.scene[sensor_path]
    vel_data = vel_sensor.data
    assert vel_data is not None
    vel_norm = torch.norm(vel_data, dim=1)  # [B]

    # Buffer last in-air velocity to approximate pre-impact speed.
    buf_key = "impact_vel_last_air"
    if buf_key not in env.extras:
      env.extras[buf_key] = torch.zeros_like(first_contact, dtype=torch.float)
    last_air_vel = env.extras[buf_key]
    in_air = found[:, idx] == 0
    last_air_vel[:, idx] = torch.where(in_air, vel_norm, last_air_vel[:, idx])

    # Use buffered velocity at touchdown; zero otherwise.
    landing_vel = torch.where(
      first_contact[:, idx], last_air_vel[:, idx], torch.zeros_like(vel_norm)
    )

    # Track velocity history to constrain speed shortly before impact.
    if use_pre_contact_window:
      window_key = "impact_vel_window_buffer"
      if window_key not in env.extras:
        env.extras[window_key] = vel_norm.new_zeros(
          (env.num_envs, len(slot_names), window_steps)
        )
      else:
        window_buf = env.extras[window_key]
        if (
          window_buf.shape[1] != len(slot_names) or window_buf.shape[2] != window_steps
        ):
          env.extras[window_key] = vel_norm.new_zeros(
            (env.num_envs, len(slot_names), window_steps)
          )
      window_buf = env.extras[window_key]
      window_buf = torch.roll(window_buf, shifts=-1, dims=2)
      window_buf[:, idx, -1] = vel_norm
      env.extras[window_key] = window_buf

      window_peak = torch.max(window_buf[:, idx, :], dim=1).values
      pre_contact_peak_vel[:, idx] = window_peak
      pre_excess = torch.clamp(window_peak - pre_contact_limit, min=0.0)
      pre_contact_cost[:, idx] = (
        torch.square(pre_excess / (pre_contact_limit + eps))
        * first_contact[:, idx].float()
      )

    # Always-on swing speed cap.
    if always_limit is not None:
      always_vel[:, idx] = vel_norm
      swing_excess = torch.clamp(vel_norm - always_limit, min=0.0)
      slot_cost = torch.square(swing_excess / (always_limit + eps))
      if always_active is not None:
        slot_cost = slot_cost * always_active
      always_cost[:, idx] = slot_cost

    # Dimensionless squared penalty: (v/limit)^2.
    cost_per_slot[:, idx] = torch.square(landing_vel / (limit + eps))
    landing_vel_per_slot[:, idx] = landing_vel

  # Sum per environment.
  cost = torch.sum(cost_per_slot, dim=1)
  if use_pre_contact_window:
    cost = cost + torch.sum(pre_contact_cost, dim=1)
  if always_limit is not None:
    cost = cost + torch.sum(always_cost, dim=1)

  # Optional gating: activate only after a given number of steps in the episode.
  if start_step > 0 and hasattr(env, "episode_length_buf"):
    active = (env.episode_length_buf >= start_step).float()
    cost = cost * active

  left_first_contact = first_contact[:, :4].float()
  right_first_contact = first_contact[:, 4:].float()
  left_landing_vel = landing_vel_per_slot[:, :4]
  right_landing_vel = landing_vel_per_slot[:, 4:]

  left_has_landing = (torch.sum(left_first_contact, dim=1) > 0).float()
  right_has_landing = (torch.sum(right_first_contact, dim=1) > 0).float()
  left_mean_per_env = torch.sum(left_landing_vel, dim=1) / torch.clamp(
    torch.sum(left_first_contact, dim=1), min=1.0
  )
  right_mean_per_env = torch.sum(right_landing_vel, dim=1) / torch.clamp(
    torch.sum(right_first_contact, dim=1), min=1.0
  )

  left_mean_landing_vel = torch.sum(left_mean_per_env * left_has_landing) / torch.clamp(
    torch.sum(left_has_landing), min=1.0
  )
  right_mean_landing_vel = torch.sum(
    right_mean_per_env * right_has_landing
  ) / torch.clamp(torch.sum(right_has_landing), min=1.0)

  foot_landings = left_has_landing + right_has_landing
  mean_landing_vel = torch.sum(
    left_mean_per_env * left_has_landing + right_mean_per_env * right_has_landing
  ) / torch.clamp(torch.sum(foot_landings), min=1.0)
  env.extras["log"]["Metrics/landing_vel_mean"] = mean_landing_vel
  env.extras["log"]["Metrics/landing_vel_left_mean"] = left_mean_landing_vel
  env.extras["log"]["Metrics/landing_vel_right_mean"] = right_mean_landing_vel

  left_marker_vel_sensor: Entity = env.scene["robot/left_foot_lin_vel"]
  right_marker_vel_sensor: Entity = env.scene["robot/right_foot_lin_vel"]
  left_marker_vel = torch.norm(left_marker_vel_sensor.data, dim=1)
  right_marker_vel = torch.norm(right_marker_vel_sensor.data, dim=1)
  env.extras["log"]["Metrics/left_foot_marker_speed"] = torch.mean(left_marker_vel)
  env.extras["log"]["Metrics/right_foot_marker_speed"] = torch.mean(right_marker_vel)
  if use_pre_contact_window:
    window_landings = torch.sum(first_contact.float())
    pre_contact_peak_at_landing = pre_contact_peak_vel * first_contact.float()
    mean_pre_contact_peak = torch.sum(pre_contact_peak_at_landing) / torch.clamp(
      window_landings, min=1
    )
    env.extras["log"]["Metrics/pre_contact_peak_vel_mean"] = mean_pre_contact_peak

  if always_limit is not None:
    vel_for_log = always_vel
    if always_active is not None:
      vel_for_log = vel_for_log * always_active.unsqueeze(1)
    env.extras["log"]["Metrics/foot_vel_max"] = torch.max(vel_for_log)

  return cost

