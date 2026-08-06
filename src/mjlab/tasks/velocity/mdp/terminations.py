from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge


class stalled_while_commanded:
  """End the episode when the robot is told to move and does not.

  Every reward edit of 2026-07-30 -- tightening the tracking kernel, removing
  the min_foot_height landing trap, restoring flat_support with a contact
  margin, adding a pre-swing weight-transfer bonus -- re-priced the motionless
  optimum without dislodging it. Run v8 converged to it anyway: root speed fell
  0.051 -> 0.017 m/s over 1200 iterations while the episodic return climbed to
  126, on 4000-step episodes that never terminated. Standing still was simply
  profitable and safe.

  Re-pricing a basin the policy is already sitting in only works if some
  gradient leads out of it. Removing the basin works regardless, which is why
  locomotion setups generally carry a termination like this one rather than
  relying on shaping alone.

  A robot is stalled when the commanded speed exceeds ``command_threshold`` and
  the measured base speed stays below ``speed_fraction`` of it continuously for
  ``grace_s``. The grace period matters: a real gait passes through
  near-zero base velocity at every stance transition, and terminating on an
  instantaneous reading would punish walking rather than standing.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.stalled_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.stalled_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    speed_fraction: float = 0.3,
    grace_s: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    commanded = torch.norm(command[:, :2], dim=1)
    speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)

    active = commanded > command_threshold
    stalled = active & (speed < speed_fraction * commanded)

    self.stalled_time = torch.where(
      stalled, self.stalled_time + env.step_dt, torch.zeros_like(self.stalled_time)
    )
    env.extras["log"]["Metrics/stalled_fraction"] = stalled.float().mean()
    return self.stalled_time > grace_s
