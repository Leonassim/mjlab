"""PD actuator with desired velocity estimated from position target changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class FiniteDifferencePdActuatorCfg(IdealPdActuatorCfg):
  """PD actuator using finite differences on position targets for desired velocity."""

  position_target_filter_alpha: float = 0.0
  """EMA coefficient for filtering the position target itself.

  Higher values keep more of the previous filtered target and reduce abrupt
  setpoint jumps seen by the PD loop. ``0.0`` means no position filtering.
  """

  velocity_target_limit: float | None = None
  """Optional clamp on the estimated desired velocity."""

  target_change_epsilon: float = 1e-6
  """Minimum target change magnitude considered as a new command."""

  velocity_target_filter_alpha: float = 0.8
  """EMA coefficient for desired velocity.

  Higher values keep more of the previous target velocity and reduce spikes from
  abrupt action changes. ``0.0`` means no filtering.
  """

  velocity_damper_di: float = 0.0
  """Inflection zone as a fraction of joint range (matches mc_rtc KinematicsConstraint
  ``diPercent``). ``0.0`` disables the velocity damper. Typical value: ``0.4``."""

  velocity_damper_ds: float = 0.0
  """Safety margin as a fraction of joint range (matches mc_rtc ``dsPercent``).
  Typical value: ``0.01``."""

  velocity_damper_vel_percent: float = 1.0
  """Fraction of nominal joint velocity limit used as max velocity in the damper
  (matches mc_rtc ``velPercent``). Typical value: ``0.9``."""

  velocity_limits: dict[str, float] | float | None = None
  """Per-joint or global velocity limit [rad/s] used by the velocity damper.
  If ``None`` and ``velocity_damper_di > 0``, the damper position projection is
  applied without a velocity clamping stage."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> FiniteDifferencePdActuator:
    return FiniteDifferencePdActuator(self, entity, target_ids, target_names)


class FiniteDifferencePdActuator(IdealPdActuator[FiniteDifferencePdActuatorCfg]):
  """Ideal PD actuator with cached desired velocity from target deltas.

  This is useful for high-gain position control where using qd_des=0 makes the
  derivative term act like pure damping and can suppress locomotion.
  """

  def __init__(
    self,
    cfg: FiniteDifferencePdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._last_position_target: torch.Tensor | None = None
    self._filtered_position_target: torch.Tensor | None = None
    self._desired_velocity_target: torch.Tensor | None = None
    self._elapsed_since_target_update: torch.Tensor | None = None
    self._initialized: torch.Tensor | None = None
    self._q_lower: torch.Tensor | None = None
    self._q_upper: torch.Tensor | None = None
    self._v_max: torch.Tensor | None = None
    # Curriculum progress [0, 1]: 0 = no damper, 1 = full mc_rtc QP constraints.
    self.velocity_damper_progress: float = 0.0

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    shape = (data.nworld, len(self._target_names))

    if self.cfg.velocity_damper_di > 0.0:
      # Read position limits from the standard MuJoCo model (jnt_range shape: (njnt, 2)).
      # mj_name2id gives the global joint ID for each target joint name.
      import mujoco as _mj

      global_ids = [
        _mj.mj_name2id(mj_model, _mj.mjtObj.mjOBJ_JOINT, n) for n in self._target_names
      ]
      self._q_lower = torch.tensor(
        [mj_model.jnt_range[jid, 0] for jid in global_ids],
        dtype=torch.float,
        device=device,
      )
      self._q_upper = torch.tensor(
        [mj_model.jnt_range[jid, 1] for jid in global_ids],
        dtype=torch.float,
        device=device,
      )

      # Velocity limits.
      vl = self.cfg.velocity_limits
      if vl is None:
        self._v_max = None
      elif isinstance(vl, (int, float)):
        self._v_max = torch.full(
          (len(self._target_names),),
          float(vl) * self.cfg.velocity_damper_vel_percent,
          device=device,
        )
      else:
        self._v_max = torch.tensor(
          [
            vl.get(n, float("inf")) * self.cfg.velocity_damper_vel_percent
            for n in self._target_names
          ],
          device=device,
        )
    self._last_position_target = torch.zeros(shape, device=device, dtype=torch.float)
    self._filtered_position_target = torch.zeros(
      shape, device=device, dtype=torch.float
    )
    self._desired_velocity_target = torch.zeros(shape, device=device, dtype=torch.float)
    self._elapsed_since_target_update = torch.zeros(
      shape, device=device, dtype=torch.float
    )
    self._initialized = torch.zeros(shape, device=device, dtype=torch.bool)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._last_position_target is not None
    assert self._filtered_position_target is not None
    assert self._desired_velocity_target is not None
    assert self._elapsed_since_target_update is not None
    assert self._initialized is not None

    uninitialized = ~self._initialized
    if torch.any(uninitialized):
      self._last_position_target = torch.where(
        uninitialized, cmd.position_target, self._last_position_target
      )
      self._filtered_position_target = torch.where(
        uninitialized, cmd.position_target, self._filtered_position_target
      )
      self._desired_velocity_target = torch.where(
        uninitialized,
        torch.zeros_like(self._desired_velocity_target),
        self._desired_velocity_target,
      )
      self._elapsed_since_target_update = torch.where(
        uninitialized,
        torch.zeros_like(self._elapsed_since_target_update),
        self._elapsed_since_target_update,
      )
      self._initialized = torch.ones_like(self._initialized)

    filtered_position_target = cmd.position_target
    pos_alpha = float(self.cfg.position_target_filter_alpha)
    if pos_alpha > 0.0:
      filtered_position_target = (
        pos_alpha * self._filtered_position_target
        + (1.0 - pos_alpha) * cmd.position_target
      )
      self._filtered_position_target = filtered_position_target

    changed = (
      torch.abs(filtered_position_target - self._last_position_target)
      > self.cfg.target_change_epsilon
    )
    if torch.any(changed):
      safe_dt = torch.clamp(self._elapsed_since_target_update, min=1e-6)
      estimated_velocity = (
        filtered_position_target - self._last_position_target
      ) / safe_dt
      if self.cfg.velocity_target_limit is not None:
        estimated_velocity = torch.clamp(
          estimated_velocity,
          -self.cfg.velocity_target_limit,
          self.cfg.velocity_target_limit,
        )
      alpha = float(self.cfg.velocity_target_filter_alpha)
      if alpha > 0.0:
        estimated_velocity = (
          alpha * self._desired_velocity_target + (1.0 - alpha) * estimated_velocity
        )
      self._desired_velocity_target = torch.where(
        changed, estimated_velocity, self._desired_velocity_target
      )
      self._last_position_target = torch.where(
        changed, filtered_position_target, self._last_position_target
      )
      self._elapsed_since_target_update = torch.where(
        changed,
        torch.zeros_like(self._elapsed_since_target_update),
        self._elapsed_since_target_update,
      )

    filtered_position_target = self._apply_velocity_damper(
      filtered_position_target, cmd.pos, cmd.vel
    )

    pd_cmd = ActuatorCmd(
      position_target=filtered_position_target,
      velocity_target=self._desired_velocity_target,
      effort_target=cmd.effort_target,
      pos=cmd.pos,
      vel=cmd.vel,
    )
    return super().compute(pd_cmd)

  def _apply_velocity_damper(
    self,
    q_target: torch.Tensor,
    q: torch.Tensor,
    qdot: torch.Tensor,
  ) -> torch.Tensor:
    """Project position target into the velocity damper safe region.

    Matches the mc_rtc KinematicsConstraint second-order velocity damper.
    When ``velocity_damper_progress`` is 0 this is a no-op; at 1 the full
    mc_rtc QP constraints are active.
    """
    p = self.velocity_damper_progress
    if p <= 0.0 or self._q_lower is None or self._q_upper is None:
      return q_target

    di = p * self.cfg.velocity_damper_di
    ds = p * self.cfg.velocity_damper_ds

    q_lo = self._q_lower  # (n_targets,)
    q_hi = self._q_upper

    range_ = q_hi - q_lo
    di_abs = di * range_
    ds_abs = ds * range_

    # Optional velocity clamp: reduce the velocity feedforward if too fast.
    if self._v_max is not None and self._desired_velocity_target is not None:
      self._desired_velocity_target = self._desired_velocity_target.clamp(
        min=-self._v_max, max=self._v_max
      )

    # Upper damper: how far above q_current the target is allowed to go.
    # alpha_hi = 1 → outside inflection zone, can reach q_hi - ds_abs
    # alpha_hi = 0 → at safety margin, can't move toward upper limit at all
    dist_hi = (q_hi - ds_abs) - q  # (n_envs, n_targets), + = away from upper limit
    span = di_abs - ds_abs  # (n_targets,)
    alpha_hi = torch.clamp(dist_hi / torch.clamp(span, min=1e-6), 0.0, 1.0)
    q_max = q + alpha_hi * dist_hi.clamp(min=0.0)

    # Lower damper: symmetric.
    dist_lo = q - (q_lo + ds_abs)
    alpha_lo = torch.clamp(dist_lo / torch.clamp(span, min=1e-6), 0.0, 1.0)
    q_min = q - alpha_lo * dist_lo.clamp(min=0.0)

    return torch.clamp(q_target, q_min, q_max)

  def update(self, dt: float) -> None:
    assert self._elapsed_since_target_update is not None
    self._elapsed_since_target_update += dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    assert self._last_position_target is not None
    assert self._filtered_position_target is not None
    assert self._desired_velocity_target is not None
    assert self._elapsed_since_target_update is not None
    assert self._initialized is not None
    self._last_position_target[env_ids] = 0.0
    self._filtered_position_target[env_ids] = 0.0
    self._desired_velocity_target[env_ids] = 0.0
    self._elapsed_since_target_update[env_ids] = 0.0
    self._initialized[env_ids] = False
