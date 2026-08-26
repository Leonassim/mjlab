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
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
)
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
  std_xy: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.

  ``std_xy`` widens the roll/pitch kernel without touching yaw. They share one
  std by default, which conflates two different things: yaw is the tracked
  command, while roll is the lateral weight transfer onto the stance hip that a
  long step needs. At std 0.35 a 0.3 rad/s roll excursion costs half the term.
  Splitting them is exactly equivalent to the single-std form when std_xy is
  None.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  if std_xy is None:
    return torch.exp(-(z_error + xy_error) / std**2)
  return torch.exp(-z_error / std**2) * torch.exp(-xy_error / std_xy**2)


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

    # mean(exp(.)) rather than the exp(mean(.)) this used to compute. With the
    # exponential outside, a single joint far from its target drove the whole
    # sum into the tail and flattened every other joint's gradient with it --
    # measured on a standing checkpoint, the term realized 0.098 of its
    # maximum while the hips sat ~0.118 rad off nominal, i.e. it had gone
    # nearly silent exactly where it was supposed to be pulling. Per-joint
    # exponentials keep each joint's own gradient alive regardless of what the
    # others are doing, and the term degrades gracefully instead of switching
    # off.
    return torch.mean(torch.exp(-error_squared / (std**2)), dim=1)

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


class gait_phase_tracking:
  """Periodic Reward Composition (Siekmann et al., "Sim-to-Real Learning of
  All Common Bipedal Gaits via Periodic Reward Composition", Cassie): an
  explicit per-env gait clock prescribes when each foot *should* be in
  swing vs stance, instead of the reactive/post-hoc approach everything
  else in this file uses (measure what happened via the contact sensor,
  reward it after the fact). air_time/min_foot_height/stride_frequency_target
  all stalled for thousands of iterations despite escalating weights --
  they only ever reward an already-discovered gait, they don't tell the
  policy what timing to try. This term does: a phase input observation
  (see mdp.gait_phase_obs) plus a dense reward for matching it gives a much
  more direct signal for gait *timing* specifically.

  Clock: phase in [0, 1) advances by (dt / period) each step, wrapped. Left
  foot uses phase directly; right foot is phase-shifted by 0.5 (the
  standard symmetric-alternating-gait assumption). Frozen (does not
  advance) while the commanded speed is below command_threshold, so
  standing still doesn't drag the clock through a swing/stance cycle with
  no visible effect -- standing_single_support_penalty already owns the
  "stand still, both feet down" case.

  Cadence is per-env and commanded-speed dependent, interpolating period
  from ``period_slow`` (at command_threshold) to ``period_fast`` (at
  ``command_ref`` and above). A fixed cadence forces step length = v*T/2 to
  collapse with v, which is precisely the "petits pas rapides" failure this
  whole line of work started from: at T=0.83s a 0.1 m/s command mechanically
  cannot produce a step longer than 4cm no matter what the rewards say.

  ``swing_duration`` is held roughly *constant* rather than scaling with the
  period (swing_ratio = swing_duration / period, so the ratio falls as the
  period grows). Scaling swing with the period instead -- i.e. a fixed
  swing_ratio -- would demand 1.0s of single support per step at the slow
  end, which this robot cannot balance, and is biomechanically backwards:
  human swing duration is near-invariant across walking speeds, and it is
  the stance/double-support fraction that stretches when you slow down.
  Since double support = 1 - 2*swing_ratio here (two feet, half-cycle
  apart), a shrinking swing_ratio at low speed *is* the growing double
  support phase, which is the desired behavior rather than a side effect.

  The policy is not given swing_ratio or the period directly: both are
  deterministic functions of the commanded velocity, which is already in
  the observation (with history). Only the phase itself needs the explicit
  channel (see mdp.gait_phase_obs), since it is integrated state the policy
  cannot recover from a single step's command.

  Also exposes ``self.amplitude`` (0 at zero command, ramping linearly to 1
  at command_threshold), consumed by mdp.gait_phase_obs to fade the phase
  observation itself to flat as the command nears zero. The frozen phase
  alone is not enough: with swing_ratio=0.5 and the 0.5 foot offset, a
  frozen phase always shows exactly one foot mid-"swing" in the sin/cos
  encoding, indistinguishable from an actively-cycling gait -- a policy
  that learned "swing encoding -> lift foot" during locomotion has no
  signal telling it that cue is stale once the clock stops (observed:
  standing_single_support_rate saturated near 100% of standing timesteps
  despite the reward already being gated to zero there).

  Per foot: reward = exp(-force^2/force_std^2) while that foot's phase is
  in the swing portion (phase < swing_ratio) -- rewards an unloaded,
  airborne foot; reward = exp(-vel^2/vel_std^2) while in the stance portion
  -- rewards a planted, non-sliding foot.

  The derived swing_ratio is clamped to <= 0.5: above that, with the two
  feet's phases a half-cycle apart, their swing windows start to overlap
  (both feet airborne at once) -- exactly what no_double_flight_penalty
  already exists to forbid, so this term would otherwise be pulling
  against it.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.phase: torch.Tensor | None = None
    self.amplitude: torch.Tensor | None = None
    self._env = env

  def reset(self, env_ids: torch.Tensor) -> None:
    if self.phase is None:
      return
    self.phase[env_ids] = torch.rand(len(env_ids), device=self.phase.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period_slow: float,
    period_fast: float,
    swing_duration: float,
    command_ref: float,
    force_std: float = 30.0,
    vel_std: float = 0.15,
    command_threshold: float = 0.1,
    stance_weight: float = 1.0,
  ) -> torch.Tensor:
    if self.phase is None:
      self.phase = torch.rand(env.num_envs, device=env.device)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_speed + angular_norm
    active = total_command > command_threshold
    self.amplitude = torch.clamp(total_command / command_threshold, 0.0, 1.0)

    # Interpolate the cycle period over the *commanded* speed range, from
    # command_threshold (where the clock starts running) to command_ref.
    speed_frac = torch.clamp(
      (total_command - command_threshold) / max(command_ref - command_threshold, 1e-6),
      0.0,
      1.0,
    )
    period = period_slow + (period_fast - period_slow) * speed_frac  # [B]
    swing_ratio = torch.clamp(swing_duration / period, max=0.5)  # [B]
    # Exposed so other terms can gate on the *prescribed* swing window rather
    # than on measured contact. See clock_swing_height_deficit.
    self.swing_ratio = swing_ratio

    self.phase = torch.where(
      active, (self.phase + env.step_dt / period) % 1.0, self.phase
    )

    phase_left = self.phase
    phase_right = (self.phase + 0.5) % 1.0
    phases = torch.stack([phase_left, phase_right], dim=1)  # [B, 2]
    should_swing = phases < swing_ratio.unsqueeze(1)  # [B, 2] bool

    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None or force.shape[1] < 8:
      raise RuntimeError("gait_phase_tracking expects 8 split foot contacts.")
    split_force = force[:, :8].view(force.shape[0], 2, 4, 3)
    foot_force_mag = torch.norm(torch.sum(split_force, dim=2), dim=-1)  # [B, 2]

    asset: Entity = env.scene[asset_cfg.name]
    foot_vel_mag = torch.norm(
      asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1
    )  # [B, 2]

    r_swing = torch.exp(-torch.square(foot_force_mag / force_std))
    r_stance = torch.exp(-torch.square(foot_vel_mag / vel_std))
    # stance_weight discounts the planted half of the cycle. A motionless robot
    # maximises r_stance (zero foot velocity) and scores zero on r_swing (both
    # feet loaded), so at equal weight the clock pays 1 - swing_ratio ~ 0.76 for
    # not walking at all -- measured at 1.65 of 2.0 on run v8's statue, its third
    # largest income. The term meant to prescribe a gait was among the reasons
    # not to have one. Discounting stance leaves the swing half, which a
    # motionless robot cannot collect, as the part worth having.
    reward = torch.where(
      should_swing, r_swing, stance_weight * r_stance
    ) * active.float().unsqueeze(1)

    # Averaged over active (moving) envs only: standing envs hold whatever
    # period the interpolation floor gives them, which would drag the mean
    # toward period_slow and hide the actual commanded-speed spread.
    active_f = active.float()
    n_active = torch.clamp(active_f.sum(), min=1.0)
    # air-time visibility moved here when the air_time reward was retired: the
    # metric was logged inside that term, and it is still the clearest read on
    # whether the clock is actually producing swings.
    current_air = sensor.data.current_air_time
    if current_air is not None and current_air.shape[1] >= 8:
      split_air = current_air[:, :8].view(current_air.shape[0], 2, 4)
      foot_in_air = 1.0 - (torch.sum(split_force.abs().sum(-1) > 0, dim=2) > 0).float()
      foot_air_time = torch.max(split_air, dim=2).values * foot_in_air
      n_air = torch.clamp(foot_in_air.sum(), min=1.0)
      env.extras["log"]["Metrics/air_time_mean"] = foot_air_time.sum() / n_air

    env.extras["log"]["Metrics/gait_period_mean"] = (period * active_f).sum() / n_active
    env.extras["log"]["Metrics/gait_phase_swing_ratio"] = (
      swing_ratio * active_f
    ).sum() / n_active
    env.extras["log"]["Metrics/gait_swing_duration_mean"] = (
      swing_ratio * period * active_f
    ).sum() / n_active
    return torch.sum(reward, dim=1)


def swing_hip_direction_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize the hip (CROTCH_P) moving opposite to the commanded
  forward/backward direction while its foot is airborne.

  Swing is hip-driven in real gait (the double-pendulum model): the hip
  pulls the thigh toward the direction of travel, and knee flexion is a
  passive/dynamic consequence of the shank trailing behind as the thigh
  accelerates -- not something to shape directly. Rewarding knee flexion, or
  penalizing any backward foot motion, would also punish that legitimate
  passive knee-bend (the foot can drift backward briefly early in swing
  purely from correct hip-driven dynamics). This only constrains the hip.

  Direction, not a fixed sign: forward command wants the hip flexing
  (negative CROTCH_P velocity in this robot's convention -- range
  [-1.92, 0.61], more negative = more flexed); backward command wants the
  opposite. cmd_vy/wz are ignored entirely (lateral/turning commands don't
  imply a sagittal hip direction to enforce), and near-zero forward command
  disables the term rather than picking an arbitrary direction.

  One-sided: driving the hip in the commanded direction, at any speed,
  costs nothing -- only the opposing direction is penalized.

  ``asset_cfg.joint_names`` must be ``(L_CROTCH_P, R_CROTCH_P)`` in that
  order (``preserve_order=True``), matching the split-foot sensor's
  (left, right) column convention.
  """
  asset: Entity = env.scene[asset_cfg.name]
  hip_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [B, 2] (left, right)

  sensor: ContactSensor = env.scene[sensor_name]
  _, foot_in_contact = _split_foot_contact_tensors(sensor)
  foot_in_air = 1.0 - foot_in_contact  # [B, 2]

  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_vx = command[:, 0]

  wrong_direction = torch.clamp(torch.sign(cmd_vx).unsqueeze(1) * hip_vel, min=0.0)
  cost = torch.sum(wrong_direction * foot_in_air, dim=1)

  active = (torch.abs(cmd_vx) > command_threshold).float()
  return cost * active


class stride_frequency_target:
  """Dense, potential-based reward for each foot's stride period (stance+
  swing) approaching a fixed target, independent of ``air_time``.

  Nothing else constrains total cycle length: ``air_time`` shapes the swing
  phase's own duration/quality, but a policy can satisfy its window with
  quick, small-amplitude steps just as well as slow, deliberate ones -- stance
  duration is unconstrained. This term taxes the full cycle directly.

  Fixed target, not speed-scaled: measured directly against a known-good
  deployed checkpoint (2026-07-10_13-52-54, replayed headless with a
  (commanded speed, period) log at every touchdown) rather than assumed.
  Result: median period was ~0.55s essentially flat across the whole
  0.05-0.5 m/s commanded range -- cadence barely depends on speed here (a
  dynamic-similarity/Froude argument predicts *some* increase with speed,
  but at these low speeds it's negligible in practice), confirming a fixed
  target is the right shape, not a speed-dependent one.

  But that ~0.55s is the *current, too-rigid* gait's cadence -- the whole
  point of this term is to move away from it, not reproduce it, so
  ``target_period`` should not just be set to the measured value. Picked
  ~0.9s instead: the leg's natural pendulum half-period (pi*sqrt(L/g) for
  this robot's 0.72m leg length) is ~0.85s, i.e. roughly the slowest swing
  a leg can do by swinging ballistically under gravity rather than actively
  braking against its own dynamics throughout -- going much slower than that
  (a 3s cycle was considered) trades "supple" for "fighting the pendulum",
  closer to tai-chi than a relaxed gait.

  First version (Gaussian kernel, charged once per touchdown) stayed
  numerically ~0 (Episode_Reward ~0.001-0.01) for 5000+ iterations: at
  period=0.4-0.6s vs target=0.9s the normalized error is 2-3 std out, and
  exp(-error^2) is negligible that far out, so the policy could not use it
  to climb toward the target from where it actually starts -- only useful
  once already close, not to get there. Two fixes:
  1. Monotonic, one-sided potential instead of a symmetric bell curve:
     Phi(t) = clamp(t, max=target_period) / target_period, t = time since
     this foot's last touchdown. Real gradient everywhere below target, flat
     (no penalty) beyond it -- same one-sided philosophy as
     ``split_feet_min_swing_height`` ("air time is free, only a low swing
     peak costs").
  2. Dense, not event-based: pays dPhi/dt = 1/target_period every step
     while t < target_period (Ng, Harada & Russell 1999), telescoping to
     Phi(period) per completed cycle -- credit arrives every step, not once
     per ~period/dt steps.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._t_since_touchdown: torch.Tensor | None = None
    self._env = env

  def reset(self, env_ids: torch.Tensor) -> None:
    if self._t_since_touchdown is not None:
      self._t_since_touchdown[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    target_period: float,
    command_threshold: float = 0.1,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    last_air_time = sensor.data.last_air_time
    last_contact_time = sensor.data.last_contact_time
    if last_air_time is None or last_contact_time is None:
      raise RuntimeError("Contact sensor must have track_air_time=True.")
    if last_air_time.shape[1] < 8:
      raise RuntimeError("Stride-frequency reward expects 8 contact slots.")

    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    split_first = first_contact[:, :8].view(first_contact.shape[0], 2, 4)
    landed = torch.any(split_first, dim=2)  # [B, 2] bool

    if self._t_since_touchdown is None or self._t_since_touchdown.shape != landed.shape:
      self._t_since_touchdown = torch.zeros_like(landed, dtype=torch.float)

    still_ramping = (self._t_since_touchdown < target_period).float()
    dense = (env.step_dt / target_period) * still_ramping

    self._t_since_touchdown = torch.where(
      landed, torch.zeros_like(self._t_since_touchdown), self._t_since_touchdown + env.step_dt
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_speed + angular_norm
    active = (total_command > command_threshold).float()
    reward = dense * active.unsqueeze(1)

    split_air = last_air_time[:, :8].view(last_air_time.shape[0], 2, 4)
    split_contact = last_contact_time[:, :8].view(last_contact_time.shape[0], 2, 4)
    foot_last_air = torch.max(split_air, dim=2).values
    foot_last_contact = torch.max(split_contact, dim=2).values
    period = foot_last_air + foot_last_contact  # [B, 2], stance+swing at touchdown
    landed_f = landed.float()
    num_landings = torch.sum(landed_f)
    env.extras["log"]["Metrics/stride_period_mean"] = torch.sum(
      period * landed_f
    ) / torch.clamp(num_landings, min=1.0)
    env.extras["log"]["Metrics/stride_period_target_mean"] = torch.tensor(
      target_period, device=period.device
    )
    return torch.sum(reward, dim=1)


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


# Open question, not a verdict: the grace period applies to
# standing_single_support alone, so for 1.5 s after every stop the reward
# demands zero velocity while hopping on one foot is free. The cheapest answer
# to that pairing is a violent stop, which is what the robot does in mc_mujoco.
#
# A ramp keyed on time-since-the-command-dropped would be hidden state: it
# scales the reward while appearing nowhere in the observation. Expose the clock
# if it is retried.

class standing_pose_penalty:
  """Pull selected joints back to the default posture at near-zero command.

  Why a separate term when ``pose`` already has a ``std_standing`` regime: that
  term cannot carry this. It returns ``mean(exp(-err^2/std^2))`` over all 30
  joints, so one joint far from nominal moves the mean by at most 1/30. At weight
  1.5 a hip yaw sitting 20 deg off nominal costs ``1.5/30 = 0.05`` -- against a
  penalty budget in the tens. That is why the splayed, duck-footed stance is free
  today: nothing that could forbid it has any leverage. Same failure mode as the
  ``feet_distance`` term realizing -0.0009 against a -10.36 budget.

  So this is a plain L2 on a *chosen* subset of joints rather than an exponential
  over all of them: the cost stays proportional to the error instead of saturating,
  and restricting the subset keeps the arms and the torso out of it (they have
  their own terms, and a stance is defined by the legs).

  Gated exactly like the rest of the standing family: a plain binary mask on the
  command, no grace window and no ramp. An earlier version scaled it by a 0->1
  clock so as not to demand the final stance mid-stride; that clock is hidden from
  the policy, which is an argument against it, but it has never been tested at a
  usable num_envs (see the StandingEngagement note above).

  Sizing: with both hips 20 deg (0.35 rad) off nominal the raw cost is
  ``2 * 0.35^2 = 0.245``, so a weight of -4 makes that stance worth about -1.0 per
  step. That is a first estimate and has never been checked on a valid run: verify
  ``Episode_Reward/standing_pose`` against the other standing terms once the robot
  walks. The -0.0008 seen on 2026-08-03 means nothing (num_envs=1, no gait).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, ids] - self.default_joint_pos[:, ids]
    cost = torch.sum(torch.square(err), dim=1)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (total_command <= command_threshold).float()

    n_standing = standing.sum().clamp(min=1.0)
    env.extras["log"]["Metrics/standing_pose_error"] = (
      torch.sqrt(cost) * standing
    ).sum() / n_standing
    return cost * standing


class standing_single_support_penalty:
  """Penalize standing on a single foot when the commanded motion is near zero.

  Skips a ``grace_period`` after each transition into the standing regime.
  Commands resample every 3-8 s (velocity_env_cfg), so an env told to stop is
  almost always mid-stride when it happens, and it physically cannot plant both
  feet until the current swing finishes -- charging that is charging an
  unavoidable transition, not a fault.

  The exemption is a hard switch. Fading it in over the window was tried on
  2026-08-03 and looked catastrophic, but those runs are void (num_envs=1); the
  question is open. See the StandingEngagement note above.

  The cost of getting this wrong scales with stride length: the conditional
  single-support-while-standing rate ran ~44% when the gait period was a fixed
  0.833 s and ~85% once the speed-dependent clock stretched it to ~1.7 s, i.e. it
  roughly doubled with the period, exactly as a fixed per-transition overhead
  would.

  ``Metrics/standing_single_support_rate`` counts only post-grace steps, so it
  measures what is actually charged. ``Metrics/standing_in_grace_rate`` counts the
  transition window separately.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.was_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.grace_left = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.was_standing[env_ids] = False
    self.grace_left[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    grace_period: float = 1.5,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    _, foot_in_contact = _split_foot_contact_tensors(sensor)  # [B, 2]
    num_feet_in_contact = torch.sum(foot_in_contact, dim=1)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = total_command <= command_threshold

    newly_standing = standing & (~self.was_standing)
    self.grace_left = torch.where(
      newly_standing,
      torch.full_like(self.grace_left, grace_period),
      torch.clamp(self.grace_left - env.step_dt, min=0.0),
    )
    self.was_standing = standing
    in_grace = standing & (self.grace_left > 0.0)
    charged = (standing & (self.grace_left <= 0.0)).float()

    one_foot = (num_feet_in_contact == 1).float()
    no_feet = (num_feet_in_contact == 0).float()
    bad = one_foot + no_feet

    # A dense height term was tried here (cost scaled by how high the
    # airborne foot is held) on the theory that the step penalty below gives
    # no direction out of the one-legged stance. Reverted 2026-07-26: over
    # 2000 iterations it moved standing_single_support_rate not at all (flat
    # at ~0.91) while roughly doubling an already-large penalty on ~40% of the
    # batch, for a behaviour the policy does not correct. fell_down tripled
    # from 0.37 to 1.01 and the run collapsed. Whatever keeps the robot on one
    # leg, it is not a missing gradient.
    cost = (one_foot + 4.0 * no_feet) * charged

    # Rate over charged steps only, i.e. what is actually paid for -- identical in
    # meaning to the pre-2026-08-03 metric, so the two are directly comparable.
    n_charged = torch.clamp(charged.sum(), min=1.0)
    n_grace = torch.clamp(in_grace.float().sum(), min=1.0)
    env.extras["log"]["Metrics/standing_single_support_rate"] = (
      bad * charged
    ).sum() / n_charged
    env.extras["log"]["Metrics/standing_in_grace_rate"] = (
      bad * in_grace.float()
    ).sum() / n_grace
    env.extras["log"]["Metrics/standing_charged_fraction"] = charged.mean()
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

    # Fraction of joint-steps whose commanded target is not executable.
    #
    # Torque saturation is not itself a defect: this is high-gain position control
    # and the robot rides its limit as normal operation. Commanding outside the
    # executable set is, and it is a learning defect first -- every command beyond
    # the window executes identically, so the policy cannot tell them apart and
    # what it learns will not hold at deployment, where mc_rtc has no clamp
    # downstream of the PD.
    #
    # Watch this rather than pd_demand_ratio_mean, which is EMA-smoothed and read
    # 3.34 on a policy whose deterministic demand was 8.45x the limit.
    instantaneous_ratio = torch.abs(demand) / torch.clamp(limit, min=1e-6)
    env.extras["log"]["Metrics/command_infeasible_fraction"] = (
      (instantaneous_ratio > 1.0).float().mean()
    )
    env.extras["log"]["Metrics/torque_demand_ratio_inst_max"] = instantaneous_ratio.max()

    # Actual forward/lateral speed of the base, ungated by command. Trivial to
    # compute and the single most decisive number this task has: on the
    # 2026-07-29 run every aggregate looked healthy (tracking error better than
    # the previous run's final, impact 2.7x lower, zero falls) while the robot
    # was translating at 0.0005 m/s against a 0.25 m/s command. It was standing
    # still, and nothing logged said so -- error_vel_xy hid it because the
    # tracking kernel was wider than the whole command range. Log the thing
    # itself, not a kernel's opinion of it.
    env.extras["log"]["Metrics/root_speed_mean"] = torch.norm(
      data.root_link_lin_vel_b[:, :2], dim=1
    ).mean()

    # Raw-action magnitude, in action units. With the scale set to the
    # actuator's own effort_limit/stiffness, one unit is one feasibility window
    # at ratio 1, so this reads directly against torque_feasibility_ratio.
    action = env.action_manager.action
    env.extras["log"]["Metrics/action_abs_mean"] = action.abs().mean()
    env.extras["log"]["Metrics/action_abs_max"] = action.abs().max()
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

  Binary gate. Fading it in over a grace window (2026-08-03) is the obvious fix
  for "be stopped NOW" and remains untested -- the runs that seemed to reject it
  had num_envs=1. See the StandingEngagement note above.
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


def standing_base_motion(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  ang_weight: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize whole-body drift of the base when the command is near zero.

  Complements ``standing_joint_vel_l2``, which is *not* enough on its own for
  the observed failure: a slow whole-body sway at standstill is essentially
  free under the current objective, for three separate reasons.

    1. ``track_linear_velocity``'s kernel is deliberately wide (std 0.40, see
       its registration comment: a tight one punishes the COM oscillation a
       long stride naturally causes). At zero command a 0.1 m/s sway costs
       ``1 - exp(-0.01/0.16)`` ~= 6% of that term -- about 0.15 out of ~2.5.
    2. ``standing_joint_vel_l2`` squares the joint velocities, so slow motion
       is quadratically discounted: it catches fast tremor, not slow drift.
    3. Nothing penalized base *translation* at all. ``body_ang_vel`` and
       ``angular_momentum`` only cover rotation.

  So this is deliberately an L1 (not L2) norm: the whole point is to keep the
  cost proportional at small velocities instead of vanishing quadratically,
  which is precisely where the existing terms stop biting. Linear and angular
  parts are summed with ``ang_weight`` scaling the angular contribution
  (rad/s and m/s are not comparable units; the default keeps a ~0.5 rad/s
  wobble worth about the same as a 0.25 m/s drift).

  Binary gate, same as ``standing_joint_vel_l2``: the grace-window fade tried on
  2026-08-03 is untested, its runs having had num_envs=1.
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_speed = torch.norm(asset.data.root_link_lin_vel_b, dim=1)
  ang_speed = torch.norm(asset.data.root_link_ang_vel_b, dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (total_command <= command_threshold).float()
  cost = (lin_speed + ang_weight * ang_speed) * standing
  n_standing = standing.sum().clamp(min=1.0)
  env.extras["log"]["Metrics/standing_base_lin_speed"] = (
    lin_speed * standing
  ).sum() / n_standing
  env.extras["log"]["Metrics/standing_base_ang_speed"] = (
    ang_speed * standing
  ).sum() / n_standing
  return cost




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
    # Latch for edge-detecting a real landing (fully airborne -> any contact).
    # See the note on spurious landings in split_feet_min_swing_height.
    self.was_in_air = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.bool
    )
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor) -> None:
    # Without this the peak tracker leaked across episode boundaries: a foot
    # airborne at termination carried its peak into the next episode's first
    # landing.
    self.peak_heights[env_ids] = 0.0
    self.was_in_air[env_ids] = False

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

    # A landing is the fully-airborne -> any-contact EDGE, not
    # ContactSensor.compute_first_contact, which fires per split slot: a foot
    # rolling heel to toe reports two or three landings per real step. Only the
    # first had ever been airborne; the rest carry peak == 0 and were charged the
    # maximum deficit, which no amount of foot lift could reduce. That is why
    # ramping this weight from -25 to -200 never moved the peak height.
    landing = self.was_in_air & (~foot_in_air)

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
    cost = torch.sum(deficit * landing.float(), dim=1) * active
    num_landings = torch.sum(landing.float())
    peak_heights_at_landing = self.peak_heights * landing.float()
    env.extras["log"]["Metrics/peak_height_mean"] = torch.sum(
      peak_heights_at_landing
    ) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landings_per_step"] = num_landings / max(
      found.shape[0], 1
    )
    self.peak_heights = torch.where(
      landing,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    self.was_in_air = foot_in_air
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


class split_feet_slip:
  """Penalize contact patches sliding across the ground.

  Tracks the world xy position of each of the eight foot contact boxes and
  charges the squared speed of any box that was touching on the previous step
  and is still touching now. That is what slipping physically is: a patch in
  contact that moves along the floor.

  Replaces a site-velocity formulation that measured the sole centre instead.
  A foot rolling over its heel or toe -- which happens on every single
  touchdown and every push-off -- swings that centre through a real velocity
  while the contact patch itself is planted, so normal gait was continuously
  billed as slipping. Per-box tracking has no such failure mode: the pivot box
  does not move, and the boxes that leave the ground stop being compared
  because the persistence mask drops them. It also localizes the fault to the
  corner that actually skidded rather than smearing it over the whole foot.

  Charging only boxes in contact on *both* steps also removes the spurious
  jump a newly-landed box would otherwise show against a stale position.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    names = [f"{s}_foot{i}_collision" for s in ("left", "right") for i in (1, 2, 3, 4)]
    geom_names = list(asset.geom_names)
    missing = [n for n in names if n not in geom_names]
    if missing:
      raise RuntimeError(f"split_feet_slip: missing foot collision geoms {missing}")
    # Same order as the split contact sensor's eight slots.
    self.geom_ids = [geom_names.index(n) for n in names]
    self.prev_xy = torch.zeros((env.num_envs, 8, 2), device=env.device)
    self.prev_contact = torch.zeros((env.num_envs, 8), device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.prev_xy[env_ids] = 0.0
    self.prev_contact[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    standing_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    found = contact_sensor.data.found
    if found is None or found.shape[1] < 8:
      raise RuntimeError("split_feet_slip expects 8 split foot contacts.")

    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = (total_command > command_threshold).float()
    scale = 1.0 + standing_scale * (1.0 - active)

    xy = asset.data.geom_pos_w[:, self.geom_ids, :2]
    contact = (found[:, :8] > 0).float()
    persist = contact * self.prev_contact
    slip_speed = torch.norm(xy - self.prev_xy, dim=-1) / env.step_dt  # [B, 8]

    cost = torch.sum(torch.square(slip_speed) * persist, dim=1) * scale

    self.prev_xy = xy
    self.prev_contact = contact

    env.extras["log"]["Metrics/slip_velocity_mean"] = torch.sum(
      slip_speed * persist
    ) / torch.clamp(torch.sum(persist), min=1.0)
    env.extras["log"]["Metrics/slip_velocity_max"] = torch.max(slip_speed * persist)
    return cost


def joint_action_acc_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Action acceleration on a named joint subset.

  Joints are named, not indexed. The action order is not stable between runs
  (policy 0 put the legs first, the current order puts them last), so a
  hardcoded index list silently penalises the wrong joints.
  """
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc[:, asset_cfg.joint_ids]), dim=1)


def stance_action_acc_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  left_asset_cfg: SceneEntityCfg,
  right_asset_cfg: SceneEntityCfg,
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
  left_acc_sq = torch.sum(torch.square(action_acc[:, left_asset_cfg.joint_ids]), dim=1)
  right_acc_sq = torch.sum(torch.square(action_acc[:, right_asset_cfg.joint_ids]), dim=1)
  return left_stance * left_acc_sq + right_stance * right_acc_sq


class action_jerk_l2:
  """Penalize action *acceleration* -- vibration -- rather than action speed.

  Smooth-but-large and small-but-shaky are opposite failure modes and the
  first difference cannot tell them apart: a clean stride ramp has a high
  action velocity and near-zero action acceleration, while a tremor has modest
  velocity and enormous acceleration. action_rate_l2, the term this replaces
  (and the single largest regularizer in the objective at ~-5.5 realized),
  penalized the first difference, so it taxed an ample stride exactly as hard
  as a jitter -- structurally unable to ask for "soft but moving".

  Also consolidates stance_action_acc_l2, upper_body_action_acc_l2 and
  leg_joint_acc_l2, which were three separate second-difference terms over
  overlapping joint subsets.

  Measured in physical joint units: the raw second difference is multiplied by
  the action scale before squaring. Raw-action penalties silently change
  meaning whenever the action scale changes -- the codebase carries several
  comments about having to rescale weights by scale^2 to compensate, and got
  the direction wrong at least once. In physical units a coefficient means the
  same thing regardless, and ``coeffs`` can then express intent (keep the
  upper body quieter than the legs) instead of correcting for units.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    import re

    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    term = env.action_manager.get_term(cfg.params.get("action_term_name", "joint_pos"))
    scale = term.scale
    n = int(env.action_manager.total_action_dim)
    if not torch.is_tensor(scale):
      scale = torch.full((n,), float(scale), device=env.device)
    scale = scale.to(env.device).reshape(-1, n)[0]  # may be broadcast per-env

    names = [asset.joint_names[i] for i in term.target_ids]
    coeffs = cfg.params.get("coeffs", {}) or {}
    w = []
    for nm in names:
      c = 1.0
      for pattern, value in coeffs.items():
        if re.search(pattern, nm):
          c = float(value)
      w.append(c)
    self._w = torch.tensor(w, device=env.device, dtype=torch.float32) * torch.square(
      scale
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    action_term_name: str = "joint_pos",
    coeffs: dict | None = None,
  ) -> torch.Tensor:
    del asset_cfg, action_term_name, coeffs
    jerk = (
      env.action_manager.action
      - 2.0 * env.action_manager.prev_action
      + env.action_manager.prev_prev_action
    )
    cost = torch.sum(self._w * torch.square(jerk), dim=1)
    env.extras["log"]["Metrics/action_jerk_mean"] = torch.sqrt(
      torch.clamp(cost, min=0.0)
    ).mean()
    return cost


class joint_torques_weighted_l2:
  """Single per-actuator torque minimisation, with per-joint emphasis.

  Replaces three overlapping terms (a flat joint_torques_l2 over every joint
  plus a dedicated ankle_roll_torque and ankle_pitch_torque on top), which
  double-taxed the ankles: they paid inside the global term *and* again in
  their own. Consolidating keeps the intent -- less torque everywhere, more
  strongly on the ankles -- while making the relative emphasis explicit and
  tunable in one place instead of emergent from three weights.

  Ankles get the larger coefficients for two reasons: they are the weakest
  actuators on the leg (45 N.m in roll, 65 N.m in pitch, against 140 at the
  hip), and a foot that needs a lot of ankle torque to stay down is a foot the
  robot placed badly -- low ankle effort and flat ground contact are the same
  good behaviour seen from two angles, so this pushes with flat_support rather
  than against it.

  ``coeffs`` maps an actuator-name regex to a multiplier; unmatched actuators
  get 1.0, and later patterns win. Duplicate ``*_motor`` entries are skipped,
  matching joint_torque_limit_margin_penalty.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    import re

    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    names = list(asset.actuator_names)
    ids = [i for i, n in enumerate(names) if not n.endswith("_motor")]
    if not ids:
      ids = list(range(len(names)))
    coeffs = cfg.params.get("coeffs", {}) or {}
    weights = []
    for i in ids:
      c = 1.0
      for pattern, value in coeffs.items():
        if re.search(pattern, names[i]):
          c = float(value)
      weights.append(c)
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self._w = torch.tensor(weights, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    coeffs: dict | None = None,
  ) -> torch.Tensor:
    del coeffs  # consumed in __init__
    asset: Entity = env.scene[asset_cfg.name]
    force = asset.data.actuator_force[:, self._ids]
    sq = torch.square(force)
    env.extras["log"]["Metrics/torque_sq_total"] = sq.sum(dim=1).mean()
    return torch.sum(self._w * sq, dim=1)


class flat_support_penalty:
  """Land flat, then stay flat -- scored as two separate things.

  Three components, all gated on the foot actually carrying weight
  (``load_threshold`` newtons; the robot weighs ~560 N, so ~280 N per foot in
  double support and ~560 N in single support, while roll-in and roll-out ramp
  through zero). Gating on bare contact instead, as this used to, charges the
  roll-through every normal step makes on its way down and again on its way
  up, and also makes lift-off itself look like a fault.

  1. **Level** -- ``deficit^2`` while loaded, where deficit is how far the
     corner count sits below ``required_contacts_per_foot``. Anchors the
     absolute standard, so landing on one corner and never improving is still
     charged.
  2. **Loss** -- any drop in corner count while loaded. Rolling onto an edge
     mid-stance is exactly the failure that hurts hardware.
  3. **Gain** -- the mirror image, credited. A foot physically cannot place
     four corners in the same instant; it touches down on one or two and
     settles. Penalising the level alone therefore taxes that unavoidable
     settling, so the descent to flat is rewarded rather than merely
     un-punished.

  Gain and loss use the same coefficient, so a full down-then-up cycle nets to
  zero and there is nothing to farm; over one loaded stance the pair telescopes
  to the net change in corner count, i.e. "end flatter than you arrived".
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self.asset_name = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    asset: Entity = env.scene[self.asset_name]
    names = [
      f"{side}_foot{i}_collision" for side in ("left", "right") for i in (1, 2, 3, 4)
    ]
    geom_names = list(asset.geom_names)
    self.geom_ids = [geom_names.index(n) for n in names]
    self.prev_count = torch.zeros((env.num_envs, 2), device=env.device)
    self.prev_loaded = torch.zeros((env.num_envs, 2), device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.prev_count[env_ids] = 0.0
    self.prev_loaded[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    required_contacts_per_foot: int = 4,
    load_threshold: float = 140.0,
    corner_tolerance: float = 0.0,
    change_gain: float = 1.0,
    standing_threshold: float = 0.1,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    force = sensor.data.force
    if found is None or force is None:
      raise RuntimeError("Contact sensor must provide 'found' and 'force'.")
    if found.shape[1] < 8:
      raise RuntimeError("flat_support_penalty expects 8 split foot contacts.")

    # Corner count by RELATIVE height, not contact detection. The sole is parallel
    # to the ground to within 16 um in the default pose, yet only 2.15 of 4 corners
    # registered: 130 um of toe-versus-heel offset per milliradian of ankle pitch
    # lifts two clear of the solver threshold, so the term measured solver luck.
    #
    # A geom margin would fix the count but change what every other contact
    # consumer sees. The four patches share a rigid body, so differences between
    # centre heights equal differences between lowest corners and tilt cancels.
    asset: Entity = env.scene[self.asset_name]
    if corner_tolerance > 0.0:
      z = asset.data.geom_pos_w[:, self.geom_ids, 2].view(found.shape[0], 2, 4)
      ref = z.min(dim=2, keepdim=True).values
      contacts = ((z - ref) < corner_tolerance).float()
    else:
      contacts = (found[:, :8] > 0).float().view(found.shape[0], 2, 4)
    contact_count = torch.sum(contacts, dim=2)  # [B, 2]
    in_contact = (contact_count > 0).float()

    foot_load = torch.norm(
      torch.sum(force[:, :8].view(force.shape[0], 2, 4, 3), dim=2), dim=-1
    )
    loaded = (foot_load > load_threshold).float()

    required = float(required_contacts_per_foot)
    deficit = torch.clamp(required - contact_count, min=0.0) / max(required, 1.0)

    # At zero command an unloaded foot is charged the FULL deficit, not nothing.
    # Charging only loaded feet lets the policy dodge this penalty by lifting a
    # foot: measured standing, both feet down costs 11.72 per step and lifting one
    # drops it to 5.86, all but cancelling standing_single_support's 6.0.
    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (total_command <= standing_threshold).float().unsqueeze(1)
    charge_mask = torch.clamp(loaded + standing * (1.0 - loaded), max=1.0)
    effective_deficit = torch.where(
      (standing > 0) & (loaded == 0), torch.ones_like(deficit), deficit
    )
    cost = torch.sum(torch.square(effective_deficit) * charge_mask, dim=1)

    # Only compare against a step that was itself loaded: the first loaded step
    # of a stance has no meaningful predecessor (the foot was airborne), and
    # crediting the 0 -> N jump there would pay for merely landing.
    both_loaded = loaded * self.prev_loaded
    delta = (contact_count - self.prev_count) / max(required, 1.0)
    lost = torch.clamp(-delta, min=0.0) * both_loaded
    gained = torch.clamp(delta, min=0.0) * both_loaded
    cost = cost + change_gain * torch.sum(lost - gained, dim=1)

    self.prev_count = contact_count
    self.prev_loaded = loaded

    env.extras["log"]["Metrics/flat_support_contacts_mean"] = torch.sum(
      contact_count * loaded
    ) / torch.clamp(torch.sum(loaded), min=1.0)
    env.extras["log"]["Metrics/stance_contacts_mean"] = torch.sum(
      contact_count * in_contact
    ) / torch.clamp(torch.sum(in_contact), min=1.0)
    env.extras["log"]["Metrics/flat_corners_lost"] = torch.sum(lost) / torch.clamp(
      torch.sum(both_loaded), min=1.0
    )
    env.extras["log"]["Metrics/loaded_foot_fraction"] = loaded.mean()
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



class preswing_weight_transfer:
  """Reward unloading the foot that is about to swing, onto the other one.

  Human gait shifts the body over the stance leg *before* the other leg leaves
  the ground. Without that transfer, lifting a foot means starting to fall
  toward the side it just left, so a policy that has not learned the transfer
  cannot lift at all -- it can only shuffle. That is exactly what the
  2026-07-29 diagnostic found: root speed 0.0005 m/s against a 0.25 m/s
  command, 1.5 deg of commanded joint amplitude, torque demand at 13% of limit
  and 0.5-1.9 rad of margin to every joint limit. Nothing physical was in the
  way; the robot had no reason and no route to unload a foot.

  This is a *directional* term, which is the point. Almost everything else
  about standing is a verdict -- flat_support, standing_single_support and the
  torque family all say "that was bad" without saying which way to move.
  ``pose`` was the only exception. This one names the next action: before the
  clock says foot i swings, get the load off foot i.

  It rides the gait clock rather than introducing a second timing source. The
  pre-swing window is the phase interval immediately before swing onset, and
  since ``gait_phase_tracking`` puts swing at phase < swing_ratio, that window
  is simply ``phase >= 1 - window``. No dependence on swing_ratio, so the term
  keeps its meaning as the clock's period and duty cycle change with speed.

  Gaming guards:
    - Normalised load *share*, not absolute force, so pressing harder with the
      stance foot is not itself rewarded.
    - Multiplied by ``support``, the total vertical load as a fraction of body
      weight. Unloading both feet at once (a hop) drives share toward 0.5 and
      support toward 0, so flight earns nothing. Without this the term would
      pay maximum during ballistic flight, which is the obvious exploit.
    - Gated by the clock's ``amplitude``, zero at zero command, so it never
      asks a standing robot to rock side to side.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    reward_name: str = "gait_phase",
    window: float = 0.15,
    body_weight: float = 560.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg
    gait = env.reward_manager.get_term_cfg(reward_name).func
    phase = getattr(gait, "phase", None)
    amplitude = getattr(gait, "amplitude", None)
    if phase is None or amplitude is None:
      return torch.zeros(env.num_envs, device=env.device)

    phases = torch.stack([phase, (phase + 0.5) % 1.0], dim=1)  # [B, 2]
    preswing = (phases >= (1.0 - window)).float()

    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    if force is None or force.shape[1] < 8:
      raise RuntimeError("preswing_weight_transfer expects 8 split foot contacts.")
    foot_force = torch.norm(
      torch.sum(force[:, :8].view(force.shape[0], 2, 4, 3), dim=2), dim=-1
    )  # [B, 2]

    total = foot_force.sum(dim=1, keepdim=True)
    share = foot_force / torch.clamp(total, min=1.0)
    support = torch.clamp(total.squeeze(1) / body_weight, max=1.0)

    # Baseline-subtracted. ``1 - 2*share`` is 0 at an even 50/50 split, 1 when
    # the pre-swing foot is fully unloaded, and clamps to 0 if that foot is
    # carrying *more* than half. Plain ``1 - share`` would have paid 0.5 for
    # standing still on two feet -- which is exactly the defect that produced
    # the statue optimum (five terms each handing out most of their value for
    # doing nothing). A term added to fix that must not reproduce it.
    transfer = (
      torch.sum(torch.clamp(1.0 - 2.0 * share, min=0.0) * preswing, dim=1)
      * support
      * amplitude
    )

    n_pre = torch.clamp(preswing.sum(), min=1.0)
    env.extras["log"]["Metrics/preswing_load_share"] = (
      (share * preswing).sum() / n_pre
    )
    return transfer


class clock_swing_height_deficit(split_feet_swing_height):
  """Charge foot-clearance shortfall over the *prescribed* swing window.

  Replaces split_feet_min_swing_height, which charged
  ``clamp(1 - peak/min_height, 0)`` once per landing, a landing being the
  transition from fully airborne to any contact. That made the penalty
  avoidable by never becoming airborne, and the 2026-07-29 perturbation test
  measured the consequence exactly: amplifying the policy's action 3x made it
  move 7x faster with zero falls, and min_foot_height alone went -0.51 ->
  -12.64, half the total reward lost. Standing still paid zero; every attempted
  step cost ~94 points at weight -100 and a 5 mm peak.

  So the term whose entire purpose was to make the robot lift its foot was the
  single largest penalty on lifting it, and raising its weight -- tried from -25
  to -200, plus two curricula -- dug the trap deeper each time. That is why
  Metrics/peak_height_mean "never responded to the weight anywhere", recorded in
  env_cfgs.py as an unexplained plateau across many runs.

  Here the window comes from the gait clock, not from contact. A foot is charged
  whenever its prescribed phase says swing, whether or not it actually left the
  ground, so:

    - standing through a prescribed swing pays the full deficit,
    - lifting 5 mm pays slightly less,
    - lifting to min_height pays nothing.

  Monotone in height and impossible to dodge by staying planted. It also removes
  the cadence exploit of the obvious alternative (paying a bonus per landing),
  since the charge follows the clock rather than the number of touchdowns.

  Gated by the clock's ``amplitude``, so a robot commanded to stand still is
  never asked to lift anything.

  Note the weight must shrink by roughly an order of magnitude relative to the
  landing-triggered version: this fires on ~25% of steps per foot instead of
  ~0.05 landings per step.
  """

  # Only getFootHeightWrtTerrain is inherited. The base class's peak tracker and
  # landing latch exist to detect touchdowns -- exactly the mechanism being
  # replaced -- so its __init__ (which demands a contact sensor) and its reset
  # are neutralised here rather than carried along as dead state.
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    min_height: float,
    reward_name: str = "gait_phase",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    gait = env.reward_manager.get_term_cfg(reward_name).func
    phase = getattr(gait, "phase", None)
    amplitude = getattr(gait, "amplitude", None)
    swing_ratio = getattr(gait, "swing_ratio", None)
    if phase is None or amplitude is None or swing_ratio is None:
      return torch.zeros(env.num_envs, device=env.device)

    phases = torch.stack([phase, (phase + 0.5) % 1.0], dim=1)  # [B, 2]
    in_swing = (phases < swing_ratio.unsqueeze(1)).float()

    heights = self.getFootHeightWrtTerrain(env, asset_cfg)  # [B, 2]
    deficit = torch.clamp(1.0 - heights / min_height, min=0.0)

    cost = torch.sum(deficit * in_swing, dim=1) * amplitude

    n_swing = torch.clamp(in_swing.sum(), min=1.0)
    env.extras["log"]["Metrics/swing_height_mean"] = (heights * in_swing).sum() / n_swing
    return cost


def raw_torque_peak_penalty(
  env: ManagerBasedRlEnv,
  soft_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Charge the peak raw PD torque of each policy step, above a band.

  ``tau_raw`` is the PD sum before MuJoCo's effort clamp, peak-held across the
  decimation window (see FiniteDifferencePdActuator._raw_torque_peak). The cost is
  ``sum_j log1p(max(0, peak_j - soft_ratio))`` on the ratio to the effort limit.

  Four deliberate choices:

  - **Peak, not mean.** What sizes an actuator is the worst instant, and with
    decimation 2 that instant can sit between two policy steps where nothing
    sampled at the policy rate would ever see it.
  - **Threshold at 1.0, the limit itself** (Leo, 2026-08-05). It was 0.7,
    inherited from pd_demand_excess where it served as an early-warning band; here
    it was an unjustified magic number. At 1.0 the threshold means something the
    hardware understands -- the motor cannot deliver this -- and the only knob left
    is the weight.
  - **log1p, not a cap.** The first version clamped the excess at 4.0, which made
    a joint at 5x and one at 128x pay exactly the same: no gradient at all on the
    peak, the same flat plateau the feasibility projection exists to remove. It
    showed up directly in the logs -- across iterations 3000-5000 of
    2026-08-04_22-xx the mean ratio fell 3.35 -> 1.93 while the max climbed
    79 -> 128. log1p keeps a gradient everywhere (5x -> 1.4, 20x -> 3.0,
    128x -> 4.8) while bounding growth, so one extreme step cannot swamp an
    episode. Two arbitrary constants removed.
  - **Summed per joint, not the max over joints.** A max would hand the gradient
    to one joint per step and leave the other 29 blind. The reported max is a
    metric, not the objective.

  Not yet done, deliberately kept for a later run so this stays one variable:
  splitting the sum into legs / arms / torso. The cost is already additive per
  joint, so no joint is arithmetically drowned -- but PPO sees one scalar return,
  so cost carried by a 13 N.m head motor adds variance to the credit assigned to
  the leg actions. Worth separating, once this shape is validated.

  Replaces the magnitude family (pd_demand_excess, torque_limit_margin,
  joint_torques_l2), which together realized about -1.81 at iteration 3500 of
  2026-08-04_09-27-15 -- that is the number to size this against. It does NOT
  replace joint_torque_rate_l2: that one charges d(tau)/dt, a different quantity,
  and it is what keeps the joints from chattering. action_jerk is not a substitute
  either, since it acts on the position target -- at a frozen action a moving
  robot still swings its torque through the kd term.
  """
  from mjlab.tasks.velocity.mdp.observations import gather_raw_torque_peak

  robot: Entity = env.scene[asset_cfg.name]
  peak = gather_raw_torque_peak(robot)
  if peak is None:
    raise RuntimeError(
      "raw_torque_peak_penalty: no actuator records _raw_torque_peak. Returning "
      "zero here would look like a satisfied constraint."
    )
  peak = peak[:, asset_cfg.joint_ids]
  excess = torch.log1p(torch.clamp(peak - soft_ratio, min=0.0))

  env.extras["log"]["Metrics/raw_torque_peak_mean"] = peak.mean()
  env.extras["log"]["Metrics/raw_torque_peak_max"] = peak.max()
  env.extras["log"]["Metrics/raw_torque_over_limit_fraction"] = (
    (peak > 1.0).float().mean()
  )
  # Which half of the PD sum carries the peak. Peak-held independently, so they do
  # not add to the total -- read them as "how big does each half get". If kd
  # dominates, this penalty is asking for slower joints, not for feasible targets.
  for _suffix, _attr in (("kp", "_raw_torque_peak_kp"), ("kd", "_raw_torque_peak_kd")):
    _parts = [
      getattr(a, _attr) for a in robot.actuators if getattr(a, _attr, None) is not None
    ]
    if _parts:
      env.extras["log"][f"Metrics/raw_torque_peak_{_suffix}_mean"] = torch.cat(
        _parts, dim=1
      ).mean()
  return torch.sum(excess, dim=1)

# --- Restored from eb2d9e4c^ for the policy 0 ablation baseline ---
# Both were deleted as dead code when their terms were dropped. Policy 0 is
# the only configuration that walked, so reproducing it needs them back.

def swing_foot_height_bonus(
  env: ManagerBasedRlEnv,
  target_height: float,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  power: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Pay, every step, for how high the swing foot actually is.

  All three existing foot-height terms are penalties measured against a target
  the gait is nowhere near -- 30 mm and 150 mm against a 3.7 mm peak. A deficit
  that large is a constant: the policy pays it whatever it does, and lifting by
  a millimetre changes it by 3%. Nothing in the budget ever paid for lifting.

  Linear by default, deliberately. The squared forms elsewhere exist to break a
  tie between two gaits that achieve the same total; here there is no tie to
  break, only a quantity that has to start moving from far below target, and a
  square would hand back a gradient of nearly zero exactly where the gait sits.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  in_air = (contact_sensor.data.found == 0).float()[:, : foot_z.shape[1]]
  # Floor-relative, same convention as observations.log_sole_height:
  # site_pos_w is absolute world z and the site sits ~20 mm up even with the
  # foot down, so scoring it raw pays a large constant for standing still and
  # leaves only a fraction of the term responding to an actual lift.
  floor = torch.quantile(foot_z.flatten(), 0.01)
  rel = torch.clamp(foot_z - floor, min=0.0)
  ratio = torch.clamp(rel / max(target_height, 1e-6), 0.0, 1.0)
  bonus = torch.sum(torch.pow(ratio, power) * in_air, dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      bonus = bonus * (total > command_threshold).float()
  env.extras["log"]["Metrics/swing_foot_lift_mean"] = torch.sum(
    rel * in_air
  ) / torch.clamp(torch.sum(in_air), min=1.0)
  return bonus


class swing_sole_clearance_bonus:
  """Pay for the clearance of the LOWEST point of the sole, not its centre.

  ``swing_foot_height_bonus`` scores one site at the middle of the sole. A foot
  can raise that site while keeping its toe on the floor simply by pitching, and
  that is what the policy learned to do: measured 6.3 cm of "lift" that walked
  into obstacles on the real robot, because the part of the foot that trips is
  the front edge and nothing was ever looking at it.

  Here the quantity is the true minimum over the whole sole. The four contact
  boxes per foot share the ankle body, so:

    lowest = min over boxes of (box centre z) - overhang(orientation)

  and the overhang of a box under rotation R is exactly ``sum_i |R[2,i]| * h_i``
  over its half-extents. Both halves matter. The min over the four centres
  catches the toe-versus-heel offset from the geometry; the overhang catches the
  box's own corner dropping below its centre, which at 20 degrees of pitch is
  17 mm on this foot -- the same order as the clearance being measured, so
  approximating it away would leave the cheat half-open.

  Tilting now buys nothing: pitching to raise the sole centre lowers the toe box
  and raises the overhang, and the term reads the worse of the two.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self.asset_name = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    asset: Entity = env.scene[self.asset_name]
    names = [
      f"{side}_foot{i}_collision" for side in ("left", "right") for i in (1, 2, 3, 4)
    ]
    geom_names = list(asset.geom_names)
    self.geom_ids = [geom_names.index(n) for n in names]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_height: float,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    power: float = 1.0,
    half_extents: tuple[float, float, float] = (0.0525, 0.0275, 0.01),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    n = asset.data.geom_pos_w.shape[0]
    z = asset.data.geom_pos_w[:, self.geom_ids, 2].view(n, 2, 4)
    # One orientation per foot: the four boxes are rigid on the same ankle body.
    quat = asset.data.geom_quat_w[:, self.geom_ids, :].view(n, 2, 4, 4)[:, :, 0, :]
    rot = matrix_from_quat(quat.reshape(-1, 4)).view(n, 2, 3, 3)
    h = torch.as_tensor(half_extents, device=z.device, dtype=z.dtype)
    overhang = torch.sum(torch.abs(rot[:, :, 2, :]) * h, dim=-1)  # [B, 2]
    lowest = z.min(dim=2).values - overhang  # [B, 2]

    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_air = (contact_sensor.data.found == 0).float()[:, : lowest.shape[1]]
    floor = torch.quantile(lowest.flatten(), 0.01)
    rel = torch.clamp(lowest - floor, min=0.0)
    ratio = torch.clamp(rel / max(target_height, 1e-6), 0.0, 1.0)
    bonus = torch.sum(torch.pow(ratio, power) * in_air, dim=1)
    if command_name is not None:
      command = env.command_manager.get_command(command_name)
      if command is not None:
        total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        bonus = bonus * (total > command_threshold).float()

    # Sole tilt, in radians, straight from the geometry. The flatness criterion
    # has been read off flat_support_contacts_mean -- how many of four contact
    # patches the solver reports loaded -- and that number measures the solver's
    # detection threshold, not the foot: this file records a sole parallel to
    # the ground within 16 um registering only 2.15 of 4. Tilt is the quantity
    # actually meant by "flat foot", it is solver-independent, and it is already
    # available here since the clearance term needs the orientation anyway.
    cos = torch.clamp(rot[:, :, 2, 2], -1.0, 1.0)
    tilt = torch.acos(cos)
    on_ground = 1.0 - in_air
    gnd = torch.clamp(torch.sum(on_ground), min=1.0)
    env.extras["log"]["Metrics/sole_tilt_loaded"] = torch.sum(tilt * on_ground) / gnd
    env.extras["log"]["Metrics/sole_tilt_swing"] = torch.sum(tilt * in_air) / torch.clamp(
      torch.sum(in_air), min=1.0
    )

    air = torch.clamp(torch.sum(in_air), min=1.0)
    env.extras["log"]["Metrics/sole_clearance_mean"] = torch.sum(rel * in_air) / air
    swing = rel[in_air > 0]
    if swing.numel() > 0:
      env.extras["log"]["Metrics/sole_clearance_p90"] = torch.quantile(swing, 0.9)
    # The cheat itself, in metres: how much the old centre-site measure
    # overstates the clearance. Near zero means the foot swings flat.
    if len(asset_cfg.site_ids) == lowest.shape[1]:
      site_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
      env.extras["log"]["Metrics/sole_height_overstated_mean"] = (
        torch.sum((site_z - lowest) * in_air) / air
      )
    return bonus


def swing_foot_height(
  env: ManagerBasedRlEnv,
  min_height: float,
  sensor_name: str | None = None,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  ramp_s: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize swing feet below min_height every step.

  When sensor_name is provided, only penalizes feet that are NOT in contact
  (i.e. in the swing phase), leaving the stance foot untouched.

  The in-air gate makes the penalty discontinuous at liftoff: a foot resting on
  the ground costs nothing, and the instant it leaves it is charged the whole
  deficit -- so attempting a short step costs more than not stepping, and
  raising the weight deepens the barrier instead of lifting the foot.
  ``ramp_s`` fades the charge in over the first ``ramp_s`` seconds of flight,
  making the term continuous. 0.0 keeps the original behaviour.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  deficit = torch.clamp(min_height - foot_z, min=0.0)  # [B, N]
  if sensor_name is not None:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_air = (contact_sensor.data.found == 0).float()  # [B, N]
    if ramp_s > 0.0:
      air_t = contact_sensor.data.current_air_time
      if air_t is None:
        raise RuntimeError("ramp_s needs a contact sensor with track_air_time=True.")
      in_air = in_air * torch.clamp(air_t[:, : in_air.shape[1]] / ramp_s, max=1.0)
    deficit = deficit * in_air
  cost = torch.sum(deficit, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def flat_touchdown_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  required_contacts_per_foot: int = 4,
  command_name: str | None = None,
  command_threshold: float = 0.0,
) -> torch.Tensor:
  """Penalize touchdowns that do not land with a flat foot.

  For each foot, if any split contact slot registers a first-contact event on the
  current step, the touchdown is considered active for that foot. The penalty is
  then based on how many of the four split contact zones are touching at that
  touchdown instant. This enforces a flat landing rather than heel-first or
  toe-first roll-over.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None:
    raise RuntimeError(
      "Contact sensor must provide 'found' for flat touchdown penalty."
    )
  if found.shape[1] < 8:
    raise RuntimeError("flat_touchdown_penalty expects 8 split foot contacts.")

  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  contacts = (found[:, :8] > 0).float().view(found.shape[0], 2, 4)
  contact_count = torch.sum(contacts, dim=2)  # [B, 2]

  touchdown = torch.stack(
    (
      torch.any(first_contact[:, :4], dim=1),
      torch.any(first_contact[:, 4:8], dim=1),
    ),
    dim=1,
  ).float()

  required_contacts = float(required_contacts_per_foot)
  deficit = torch.clamp(required_contacts - contact_count, min=0.0) / max(
    required_contacts, 1.0
  )
  cost = torch.sum(torch.square(deficit) * touchdown, dim=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    cost = cost * (total_command > command_threshold).float()

  env.extras["log"]["Metrics/flat_touchdown_contacts_mean"] = torch.sum(
    contact_count * touchdown
  ) / torch.clamp(torch.sum(touchdown), min=1.0)
  # Impulse, paid only on the touchdown step, so RewardManager's dt scaling
  # divides it by ~50 -- the same defect that left com_step_progress inert.
  # Divide it back out so `weight` means what it means for every other term.
  return cost / env.step_dt

def is_terminated_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Termination as a per-second rate, so the weight means something.

  mdp.is_terminated returns a 0/1 impulse and RewardManager multiplies by dt,
  so at -2000 the realized cost of an 8% fall rate was -0.004: four hundredths
  of one percent of the negative budget. Falls were free. Third instance of the
  same defect after com_step_progress and flat_touchdown.
  """
  return env.termination_manager.terminated.float() / env.step_dt


class direction_progress:
  """Pay for going *where* the command points, not for holding its exact vector.

  ``track_linear_velocity`` scores exp(-|c - v|^2 / std^2) on the raw
  instantaneous velocity, with std 0.20. A real step is oscillatory in all
  three components -- it decelerates at heel strike, accelerates at push-off,
  the CoM rises and falls, and the weight swings onto the stance hip. Every one
  of those excursions reads as tracking error under an isotropic kernel that
  narrow. The only gait holding v near-constant is a shuffle, so the term
  prices the long step out before any other reward gets a say.

  Here the velocity is low-passed over roughly one gait cycle first, so
  intra-step oscillation is invisible. What is left is scored anisotropically:

    along the command  linear ramp to 1.0 at the commanded speed, then flat --
                       going faster is never punished, going slower degrades
                       gracefully instead of falling off a narrow exponential
    across it          wide kernel, enough to stop crabbing, not enough to
                       forbid the lateral weight shift a step needs
    vertical           dropped; CoM rise and fall is part of stepping

  Below ``command_threshold`` the term pays for standing still instead, on the
  same wide kernel, so stand-still stability is unchanged.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.v_f = torch.zeros(env.num_envs, 2, device=env.device)
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.v_f.zero_()
    else:
      self.v_f[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    tau: float = 0.4,
    lateral_std: float = 0.35,
    standing_std: float | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    v = asset.data.root_link_lin_vel_b[:, :2]
    a = self.step_dt / max(tau, self.step_dt)
    self.v_f = self.v_f + a * (v - self.v_f)

    lin = command[:, :2]
    speed = torch.norm(lin, dim=1)
    heading = lin / speed.clamp(min=1e-6).unsqueeze(1)
    along = (self.v_f * heading).sum(dim=1)
    across = self.v_f[:, 0] * (-heading[:, 1]) + self.v_f[:, 1] * heading[:, 0]

    progress = torch.clamp(along / speed.clamp(min=1e-6), 0.0, 1.0)
    on_axis = torch.exp(-torch.square(across) / lateral_std**2)
    # Standing gets its own, tighter kernel. Sharing lateral_std made standing
    # far more permissive than policy 0, which held still under
    # track_linear_velocity at std 0.20: at 0.1 m/s of drift policy 0 lost 22%
    # of the term and this lost 8%.
    s_std = lateral_std if standing_std is None else standing_std
    still = torch.exp(-torch.sum(torch.square(self.v_f), dim=1) / s_std**2)
    moving = (speed + torch.abs(command[:, 2])) > command_threshold

    env.extras["log"]["Metrics/vel_sway_rms"] = torch.sqrt(
      torch.mean(torch.square(v[:, 1] - self.v_f[:, 1]))
    )
    return torch.where(moving, progress * on_axis, still)


class com_step_progress:
  """Pay for the ground covered *per step*, not per second.

  Velocity tracking cannot tell two gaits apart: short fast steps and long slow
  ones average the same speed and collect the same reward, so the policy picks
  whichever is cheaper elsewhere -- which is the shuffle. Rewarding CoM
  displacement does not fix that, because displacement per unit time is the
  velocity again.

  What breaks the tie is paying at touchdown, superlinearly. Two half-length
  steps pay 2*(0.5)^power and one full step pays 1.0, so at power 2 the same
  distance in the same time earns twice as much when it is taken in one stride.
  Same structure as split_feet_air_time's square, on the quantity actually
  wanted -- distance -- instead of the duration that only stood in for it.

  Unlike slowing the cadence, this asks nothing of single-support balance: the
  flight time can stay where the plant allows it.

  The accumulator integrates base velocity projected on the commanded heading,
  so it measures progress *along the command* and ignores drift across it.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.accum = torch.zeros(env.num_envs, device=env.device)
    self.elapsed = torch.zeros(env.num_envs, device=env.device)
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.accum.zero_()
      self.elapsed.zero_()
    else:
      self.accum[env_ids] = 0.0
      self.elapsed[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    target_distance: float = 0.10,
    target_period: float = 0.0,
    power: float = 2.0,
    command_threshold: float = 0.1,
    min_air_time: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    lin = command[:, :2]
    speed = torch.norm(lin, dim=1)
    heading = lin / speed.clamp(min=1e-6).unsqueeze(1)
    v_body = asset.data.root_link_lin_vel_b[:, :2]
    self.accum = self.accum + (v_body * heading).sum(dim=1) * self.step_dt
    self.elapsed = self.elapsed + self.step_dt

    # Debounce. compute_first_contact fires on contact chatter too: measured
    # 9.06 events/s against the ~4.8 a two-foot gait at air_time 0.165 can
    # produce, so roughly every footfall was counted twice. That paid the term
    # twice over AND reset the accumulator mid-step, halving the step length it
    # reports. A landing counts only after a real flight phase.
    sensor: ContactSensor = env.scene[sensor_name]
    first = sensor.compute_first_contact(dt=self.step_dt)
    air = sensor.data.last_air_time
    if air is not None:
      first = first & (air[:, : first.shape[1]] > min_air_time)
    landed = first.any(dim=1)
    ratio = torch.clamp(self.accum / max(target_distance, 1e-6), 0.0, 1.0)
    scored = torch.pow(ratio, power)
    if target_period > 0.0:
      # Distance alone does not slow the gait: at a given speed the policy can
      # collect the same distance in fast small steps. Pay for the period too,
      # averaged rather than multiplied so one lagging half does not zero the
      # term. Stance time counts, so the cadence can slow by standing longer --
      # which is the cheap half of "one real ample slow step".
      p_ratio = torch.clamp(self.elapsed / target_period, 0.0, 1.0)
      scored = 0.5 * (scored + torch.pow(p_ratio, power))
    active = (speed + torch.abs(command[:, 2]) > command_threshold).float()
    reward = scored * landed.float() * active

    n = torch.clamp(torch.sum(landed.float()), min=1.0)
    env.extras["log"]["Metrics/step_length_mean"] = (
      torch.sum(self.accum * landed.float()) / n
    )
    env.extras["log"]["Metrics/step_rate"] = torch.mean(landed.float()) / self.step_dt
    env.extras["log"]["Metrics/step_period_mean"] = (
      torch.sum(self.elapsed * landed.float()) / n
    )
    self.accum = torch.where(landed, torch.zeros_like(self.accum), self.accum)
    self.elapsed = torch.where(landed, torch.zeros_like(self.elapsed), self.elapsed)

    # RewardManager returns raw * weight * dt: every term is a per-second rate.
    # An impulse paid once per touchdown gets that dt too, which divided this
    # term by ~50 and left it at 0.1% of the positive budget -- inert. Divide it
    # back out so `weight` means what it means everywhere else, and the term is
    # worth weight * (landings per second) at full stride.
    return reward / self.step_dt


class sole_flat_touchdown_bonus:
  """Pay for a level sole at the instant it lands.

  Leo's hardware verdict on the high-clearance policy: it trips early because
  the feet are tilted in the air. The foot that trips is the one that arrives
  edge-first, so the quantity is the tilt at first contact -- not the tilt once
  loaded, which the ground has already corrected, and not the corner count,
  which measures the solver's contact threshold rather than the foot.

  A bonus, not a penalty. Every landing demand raised as a cost on this gait was
  answered by the policy not landing: flat_touchdown x2.7, impact_vel x3 and the
  0.20 -> 0.16 threshold each produced more falls than the change they asked
  for. Paying for the good landing leaves no cheaper escape than doing it.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self.asset_name = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name
    asset: Entity = env.scene[self.asset_name]
    names = [
      f"{side}_foot{i}_collision" for side in ("left", "right") for i in (1, 2, 3, 4)
    ]
    geom_names = list(asset.geom_names)
    self.geom_ids = [geom_names.index(n) for n in names]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    scale: float = 0.06,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    n = asset.data.geom_pos_w.shape[0]
    quat = asset.data.geom_quat_w[:, self.geom_ids, :].view(n, 2, 4, 4)[:, :, 0, :]
    rot = matrix_from_quat(quat.reshape(-1, 4)).view(n, 2, 3, 3)
    tilt = torch.acos(torch.clamp(rot[:, :, 2, 2], -1.0, 1.0))  # [B, 2]

    contact_sensor: ContactSensor = env.scene[sensor_name]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt).float()
    first_contact = first_contact[:, : tilt.shape[1]]

    bonus = torch.exp(-torch.square(tilt / max(scale, 1e-6))) * first_contact
    bonus = torch.sum(bonus, dim=1)
    if command_name is not None:
      command = env.command_manager.get_command(command_name)
      if command is not None:
        total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        bonus = bonus * (total > command_threshold).float()

    land = torch.clamp(torch.sum(first_contact), min=1.0)
    env.extras["log"]["Metrics/sole_tilt_touchdown"] = (
      torch.sum(tilt * first_contact) / land
    )
    return bonus
