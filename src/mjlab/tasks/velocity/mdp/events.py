"""Randomisation events specific to the velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_actuator_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stiffness_range: tuple[float, float] = (0.9, 1.1),
  damping_range: tuple[float, float] = (0.9, 1.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale kp and kd by a factor drawn per environment and per joint.

  Always drawn from ``default_stiffness`` / ``default_damping``, never from the
  current value: factors would otherwise compound from one episode to the next.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  for act in asset.actuators:
    stiffness = getattr(act, "stiffness", None)
    damping = getattr(act, "damping", None)
    default_k = getattr(act, "default_stiffness", None)
    default_d = getattr(act, "default_damping", None)
    if stiffness is None or damping is None:
      continue
    if default_k is None or default_d is None:
      raise RuntimeError(
        f"randomize_actuator_gains: l'actionneur {type(act).__name__} n'expose "
        "pas default_stiffness/default_damping"
      )
    shape = (len(env_ids), stiffness.shape[1])
    fk = torch.empty(shape, device=stiffness.device).uniform_(*stiffness_range)
    fd = torch.empty(shape, device=damping.device).uniform_(*damping_range)
    stiffness[env_ids] = default_k[env_ids] * fk
    damping[env_ids] = default_d[env_ids] * fd


def randomize_posture_task_stiffness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  stiffness_range: tuple[float, float] = (0.75, 1.25),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomise the stiffness of the PostureTask filter modelled in training.

  Randomising K randomises the delay the policy sees -- the only delay modelled
  here. Damping follows as 2*sqrt(K) in the actuator, so this is one plant, not
  two independent gains. Drawn from the stored default, never from the current
  value.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  touched = False
  for act in asset.actuators:
    stiffness = getattr(act, "posture_stiffness", None)
    default = getattr(act, "default_posture_stiffness", None)
    if stiffness is None or default is None:
      continue
    factor = torch.empty(
      (len(env_ids), stiffness.shape[1]), device=stiffness.device
    ).uniform_(*stiffness_range)
    stiffness[env_ids] = default[env_ids] * factor
    touched = True

  if not touched:
    # Staying silent would read as success in the logs while randomising nothing.
    raise RuntimeError(
      "randomize_posture_task_stiffness: no actuator exposes posture_stiffness. "
      "Is the PostureTask filter configured (posture_task_stiffness)?"
    )


def randomize_sensor_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bias_ranges: dict[str, tuple[float, float]] | None = None,
  scale_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
  """Draw a sensor bias held constant over an episode, per environment.

  Distinct from observation noise, which is re-centred every step: a five-frame
  history averages zero-mean noise away but stays credulous to an offset.

  Keys must match what ``observations.base_lin_vel_biased`` and
  ``projected_gravity_biased`` read; any other observation term ignores the bias
  silently, hence the validation below.
  """
  known = {"base_lin_vel": 3, "projected_gravity": 3, "base_ang_vel": 3}
  if bias_ranges is None:
    bias_ranges = {}
  if scale_ranges is None:
    scale_ranges = {}
  unknown = (set(bias_ranges) | set(scale_ranges)) - set(known)
  if unknown:
    raise ValueError(
      f"randomize_sensor_bias: unknown keys {sorted(unknown)}. "
      f"Expected among {sorted(known)}."
    )

  store: dict[str, torch.Tensor] = getattr(env, "_rhps1_sensor_bias", {})
  if not hasattr(env, "_rhps1_sensor_bias"):
    env._rhps1_sensor_bias = store

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  for key, (lo, hi) in bias_ranges.items():
    dim = known[key]
    tensor = store.get(key)
    if tensor is None:
      tensor = torch.zeros((env.num_envs, dim), device=env.device)
      store[key] = tensor
    tensor[env_ids] = torch.empty(
      (len(env_ids), dim), device=env.device
    ).uniform_(lo, hi)

  # Relative error, applied as value * (1 + scale). An estimator whose output is
  # filtered to zero at standstill has a gain error, not an offset: an additive
  # bias would make it report motion while the robot is still.
  sstore: dict[str, torch.Tensor] = getattr(env, "_rhps1_sensor_scale", {})
  if not hasattr(env, "_rhps1_sensor_scale"):
    env._rhps1_sensor_scale = sstore
  for key, (lo, hi) in scale_ranges.items():
    dim = known[key]
    tensor = sstore.get(key)
    if tensor is None:
      tensor = torch.zeros((env.num_envs, dim), device=env.device)
      sstore[key] = tensor
    tensor[env_ids] = torch.empty(
      (len(env_ids), dim), device=env.device
    ).uniform_(lo, hi)
