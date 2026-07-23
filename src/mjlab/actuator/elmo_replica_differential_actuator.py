"""Real RHPS1 ELMO drive replica for a differentially-coupled joint pair.

5 of the 22 ACTUATOR_TYPE_ROTATE joint groups on RHPS1 are NOT one motor per
joint: two physical drives ("channels") each run their own P(pos)/PI(vel)
loop, but the ANGLE each channel tracks is a fixed +/-1 sum/difference mix of
the two anatomical joint angles, not either joint's angle directly. This is
hardcoded on the real robot in rhps1-iob/TransListenerEx.hpp
(`convert_angle_hrpsys_vec_to_shm`), e.g. for the shoulder:

    buffer[L_SHOULDER_RYF] = r + y;   // r = L_SHOULDER_R, y = L_SHOULDER_Y
    buffer[L_SHOULDER_RYB] = r - y;

By virtual work, joint torque is the SAME linear combination of the two
channels' own torques (each computed exactly like ElmoReplicaActuator, with
its own Kp_pos/Kp_vel/Ki_vel/N/Kt/current-limit -- these often differ from its
partner channel, e.g. R_ELBOW's two channels have different Kp_pos entirely):

    tau_dof_a = tau_channel_a + tau_channel_b
    tau_dof_b = tau_channel_a - tau_channel_b

The 5 pairs (10 joints) this covers, with (dof_a, dof_b, channel_a, channel_b)
such that channel_a = dof_a + dof_b and channel_b = dof_a - dof_b (see
RHPS1_gains/README.md "Torque mixing for the 5 differential joint pairs" for
the full derivation from TransListenerEx.hpp):

    CHEST:        dof_a=CHEST_P,        dof_b=CHEST_Y,        ch_a=ChestYPL,     ch_b=ChestYPR
    HEAD:         dof_a=HEAD_P,         dof_b=HEAD_Y,         ch_a=HeadYPL,      ch_b=HeadYPR
    L_SHOULDER:   dof_a=L_SHOULDER_R,   dof_b=L_SHOULDER_Y,   ch_a=LShoulderRYF, ch_b=LShoulderRYB
    R_SHOULDER:   dof_a=R_SHOULDER_R,   dof_b=R_SHOULDER_Y,   ch_a=RShoulderRYF, ch_b=RShoulderRYB
    L_ELBOW:      dof_a=L_ELBOW_P,      dof_b=L_ELBOW_Y,      ch_a=LElbowPYO,    ch_b=LElbowPYI
    R_ELBOW:      dof_a=R_ELBOW_P,      dof_b=R_ELBOW_Y,      ch_a=RElbowPYI,    ch_b=RElbowPYO
    L_WRIST:      dof_a=L_WRIST_R,      dof_b=L_WRIST_Y,      ch_a=LWristPYI,    ch_b=LWristPYO
    R_WRIST:      dof_a=R_WRIST_R,      dof_b=R_WRIST_Y,      ch_a=RWristPYI,    ch_b=RWristPYO

Note L_ELBOW and R_ELBOW have opposite channel-to-sign assignment (mirrored
hardware) -- double-checked against TransListenerEx.hpp, not a typo.

NOT wired into any robot config yet -- see elmo_replica_actuator.py's module
docstring and the conversation summary handed to the other Claude Code
session. Same limitations apply (constant current cap, no 8s peak->continuous
derate, eta calibrated not measured, no back-EMF, anti-windup unverified).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.actuator.elmo_replica_actuator import RAD_TO_COUNT
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class ElmoChannelParams:
  """One physical drive's real P(pos)/PI(vel)/current-limit parameters.

  Same fields as ElmoReplicaActuatorCfg, factored out so a differential pair
  (two channels, generally different parameters -- see module docstring) can
  hold two of these instead of duplicating the actuator-cfg machinery.
  """

  Kp_pos: float
  Kp_vel: float
  Ki_vel: float
  gear_ratio: float
  torque_constant: float
  current_limit_continuous: float
  current_limit_peak: float | None = None
  eta: float = 1.0

  def __post_init__(self) -> None:
    assert self.Kp_pos > 0 and self.Kp_vel > 0 and self.Ki_vel > 0
    assert self.gear_ratio > 0 and self.torque_constant > 0
    assert self.current_limit_continuous > 0
    assert 0.0 < self.eta <= 1.0

  @property
  def tau_max_continuous(self) -> float:
    return self.eta * self.gear_ratio * self.torque_constant * self.current_limit_continuous

  @property
  def tau_max_peak(self) -> float | None:
    if self.current_limit_peak is None:
      return None
    return self.eta * self.gear_ratio * self.torque_constant * self.current_limit_peak


@dataclass(kw_only=True)
class ElmoReplicaDifferentialActuatorCfg(ActuatorCfg):
  """Two-channel differential real-drive replica for one anatomical joint pair.

  `target_names_expr` must list exactly (dof_a_name, dof_b_name) in that
  order, matching `channel_a`/`channel_b` such that channel_a tracks
  (dof_a + dof_b) and channel_b tracks (dof_a - dof_b) -- see module
  docstring for the per-pair (dof_a, dof_b, channel_a, channel_b) table.
  """

  channel_a: ElmoChannelParams
  channel_b: ElmoChannelParams

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> ElmoReplicaDifferentialActuator:
    assert len(target_names) == 2, (
      f"ElmoReplicaDifferentialActuatorCfg expects exactly 2 targets (dof_a, dof_b), "
      f"got {target_names}"
    )
    return ElmoReplicaDifferentialActuator(self, entity, target_ids, target_names)


class ElmoReplicaDifferentialActuator(Actuator[ElmoReplicaDifferentialActuatorCfg]):
  """Runs two independent ElmoReplicaActuator-style channel loops on the
  +/-1 sum/difference of the pair's two joint angles, then combines the two
  channel torques back into per-joint torques by the same +/-1 mix (virtual
  work). See module docstring.

  Internal per-channel state (Iv, last_ev, last_u, last_u_sat) has shape
  (num_envs, 2): column 0 = channel_a, column 1 = channel_b. dof_a is always
  target index 0, dof_b index 1 (asserted in build()).
  """

  def __init__(
    self,
    cfg: ElmoReplicaDifferentialActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.Iv: torch.Tensor | None = None
    self._last_ev: torch.Tensor | None = None
    self._last_u: torch.Tensor | None = None
    self._last_u_sat: torch.Tensor | None = None
    # (2,) tensors of per-channel constants, built once in initialize().
    self._Kp_pos: torch.Tensor | None = None
    self._Kp_vel: torch.Tensor | None = None
    self._Ki_vel: torch.Tensor | None = None
    self._scale: torch.Tensor | None = None  # gear_ratio * RAD_TO_COUNT, per channel
    self._tau_gain: torch.Tensor | None = None  # eta * gear_ratio * torque_constant
    self._i_limit: torch.Tensor | None = None  # current_limit_continuous

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    dof_a_name, dof_b_name = target_names
    # MuJoCo-side forcerange backstop per joint = sum of both channels' peak (or
    # continuous) torque -- the worst case if both channels saturate the same way.
    a, b = self.cfg.channel_a, self.cfg.channel_b
    backstop = (a.tau_max_peak or a.tau_max_continuous) + (
      b.tau_max_peak or b.tau_max_continuous
    )
    for name in (dof_a_name, dof_b_name):
      actuator = create_motor_actuator(
        spec,
        name,
        effort_limit=backstop,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        viscous_damping=self.cfg.viscous_damping,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    shape = (data.nworld, 2)
    self.Iv = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_ev = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_u = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_u_sat = torch.zeros(shape, device=device, dtype=torch.float)

    a, b = self.cfg.channel_a, self.cfg.channel_b
    self._Kp_pos = torch.tensor([a.Kp_pos, b.Kp_pos], device=device)
    self._Kp_vel = torch.tensor([a.Kp_vel, b.Kp_vel], device=device)
    self._Ki_vel = torch.tensor([a.Ki_vel, b.Ki_vel], device=device)
    self._scale = torch.tensor(
      [a.gear_ratio * RAD_TO_COUNT, b.gear_ratio * RAD_TO_COUNT], device=device
    )
    self._tau_gain = torch.tensor(
      [
        a.eta * a.gear_ratio * a.torque_constant,
        b.eta * b.gear_ratio * b.torque_constant,
      ],
      device=device,
    )
    self._i_limit = torch.tensor(
      [a.current_limit_continuous, b.current_limit_continuous], device=device
    )

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.Iv is not None
    assert self._scale is not None
    assert self._Kp_pos is not None and self._Kp_vel is not None and self._Ki_vel is not None
    assert self._tau_gain is not None and self._i_limit is not None

    # cmd.* columns are [dof_a, dof_b] (target_names order, asserted in build()).
    # Channel mix: channel_a = dof_a + dof_b, channel_b = dof_a - dof_b.
    mix = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=cmd.pos.device)  # (2,2): rows=channels

    ch_pos_target = cmd.position_target @ mix.T  # (num_envs, 2) -> channel-space targets
    ch_pos = cmd.pos @ mix.T
    ch_vel_ff = cmd.velocity_target @ mix.T
    ch_vel = cmd.vel @ mix.T

    q_ref_count = ch_pos_target * self._scale
    q_count = ch_pos * self._scale
    eq_count = q_ref_count - q_count
    v_ff_count = ch_vel_ff * self._scale
    v_ref_count = self._Kp_pos * eq_count + v_ff_count

    qdot_count = ch_vel * self._scale
    ev = v_ref_count - qdot_count

    u = self._Kp_vel * ev + self._Ki_vel * self.Iv
    u_sat = torch.clamp(u, -self._i_limit, self._i_limit)
    tau_channel = self._tau_gain * u_sat  # (num_envs, 2): [tau_ch_a, tau_ch_b]

    self._last_ev = ev
    self._last_u = u
    self._last_u_sat = u_sat

    # tau_dof_a = tau_ch_a + tau_ch_b, tau_dof_b = tau_ch_a - tau_ch_b -- same +/-1
    # mix as the angle mixing above (virtual work on a +/-1 linear map is self-adjoint).
    tau_dof = tau_channel @ mix
    return tau_dof

  def update(self, dt: float) -> None:
    assert self.Iv is not None
    assert self._last_ev is not None
    assert self._last_u is not None
    assert self._last_u_sat is not None
    assert self._i_limit is not None

    is_saturating = self._last_u.abs() > self._i_limit
    winding_up = is_saturating & (self._last_ev * self._last_u_sat > 0.0)
    d_iv = torch.where(winding_up, torch.zeros_like(self._last_ev), self._last_ev)
    self.Iv = self.Iv + d_iv * dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    assert self.Iv is not None
    assert self._last_ev is not None
    assert self._last_u is not None
    assert self._last_u_sat is not None
    self.Iv[env_ids] = 0.0
    self._last_ev[env_ids] = 0.0
    self._last_u[env_ids] = 0.0
    self._last_u_sat[env_ids] = 0.0
