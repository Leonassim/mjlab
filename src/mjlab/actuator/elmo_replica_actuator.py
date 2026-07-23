"""Real RHPS1 ELMO drive replica: cascaded P(pos)/PI(vel) loop with current
saturation and anti-windup.

NOT wired into any robot config yet (see rhps1_constants.py: the
RHPS1_ELMO_ACTUATOR_* cfgs below exist but are not in RHPS1_ACTUATORS). Written,
not activated -- see the conversation summary handed to the other Claude Code
session for why.

Ports the same model as RHPS1_gains/pdgains/elmo_joint_controller.py (Python,
validated against the notebook's simulate_closed_loop bit-for-bit off the
saturation boundary) and RHPS1_gains's rl_controller/src/ElmoJointReplica.h
(C++, wired into the mc_rtc NewRLQPController for live logging in mc_mujoco) to
this repo's batched (num_envs, num_targets) torch/Actuator framework:

  e_pos = q_ref_count - q_count
  v_ref_count = Kp_pos * e_pos + v_ff_count
  e_vel = v_ref_count - qdot_count
  i_cmd = Kp_vel * e_vel + Ki_vel * integral(e_vel), saturated to +/-i_limit,
          with anti-windup (integral freezes while saturated and the error
          would deepen it)
  tau_joint = eta * N * Kt * i_cmd

Only meaningful for ACTUATOR_TYPE_ROTATE joints (the 22 in
RHPS1_gains/README.md's "Actuator Limits" table: both knees, crotch_Y, chest,
head, shoulders, elbows, wrists) -- NOT the 8 linear-actuator joints (hip
roll/pitch, ankle roll/pitch), which use a parallel-cylinder mechanism this
class does not model (missing cylinder attachment-point geometry, see
RHPS1_gains/README.md).

Known limitations, carried over from the Python/C++ ports:
  - i_limit is a single constant (uses the drive's continuous CL[1] rating).
    The real drive tolerates a higher peak (PL[1]) for up to PL[2] (~8s) before
    derating to CL[1] -- that two-tier, time-based dynamic is NOT modeled here.
  - eta (transmission efficiency) is a calibrated estimate (~0.77 for the
    knees, from two independent cross-checks), not a measurement. Defaults to
    1.0 (upper bound) unless set explicitly; treat any value here as a
    starting point to refine, not ground truth.
  - No back-EMF / torque-speed derating (tau_max is constant regardless of
    qdot).
  - Anti-windup is a standard industrial-servo assumption, NOT verified from
    the ELMO Gold-line docs for this specific drive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity

COUNTS_PER_REV = 65536.0
RAD_TO_COUNT = COUNTS_PER_REV / (2.0 * math.pi)


@dataclass(kw_only=True)
class ElmoReplicaActuatorCfg(ActuatorCfg):
  """Real ELMO drive P(pos)/PI(vel) cascaded loop with current saturation.

  One instance per physical drive/joint is expected (real Kp_pos/Kp_vel/Ki_vel/
  N/Kt/current-limits are joint-specific, not shared even L/R -- see
  RHPS1_gains/FromRealRobot/drive_gains_map.csv), unlike the flat
  IdealPdActuatorCfg-family configs elsewhere in this file which apply one
  stiffness/damping/effort_limit to a whole `target_names_expr` group.
  """

  Kp_pos: float
  """Position loop gain (KP3 in drive_gains_map.csv). Units: 1/s-ish (rad error * N *
  RAD_TO_COUNT -> counts/s velocity target)."""

  Kp_vel: float
  """Velocity loop proportional gain (KP2). Units: A/(count/s)."""

  Ki_vel: float
  """Velocity loop integral gain (KI2). Units: A/(count/s) per second, i.e. Hz * KP2-like."""

  gear_ratio: float
  """N: motor revolutions per joint revolution. Measured empirically (real robot has no
  datasheet/URDF value for this), NOT a nominal/design ratio."""

  torque_constant: float
  """Kt [Nm/Arms]: motor torque constant, from the SANMOTION datasheet mapping."""

  current_limit_continuous: float
  """CL[1] [A]: the drive's continuous current limit, from its own .gprm. Used as the
  (constant) saturation bound -- see module docstring for what this simplifies away."""

  current_limit_peak: float | None = None
  """PL[1] [A]: the drive's peak current limit, informational only here (used for the
  MuJoCo actuator's own forcerange backstop, not for the running saturation, which always
  uses current_limit_continuous)."""

  eta: float = 1.0
  """Transmission efficiency. NOT measured -- see module docstring. 1.0 makes tau_max an
  upper bound; calibrated estimates should be passed in explicitly per joint."""

  def __post_init__(self) -> None:
    super().__post_init__()
    assert self.Kp_pos > 0 and self.Kp_vel > 0 and self.Ki_vel > 0
    assert self.gear_ratio > 0 and self.torque_constant > 0
    assert self.current_limit_continuous > 0
    assert 0.0 < self.eta <= 1.0

  @property
  def tau_max_continuous(self) -> float:
    """eta-scaled upper bound on continuously-sustainable joint torque [Nm]."""
    return self.eta * self.gear_ratio * self.torque_constant * self.current_limit_continuous

  @property
  def tau_max_peak(self) -> float | None:
    """eta-scaled upper bound on short-burst (<= PL[2], typically ~8s) joint torque [Nm]."""
    if self.current_limit_peak is None:
      return None
    return self.eta * self.gear_ratio * self.torque_constant * self.current_limit_peak

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> ElmoReplicaActuator:
    return ElmoReplicaActuator(self, entity, target_ids, target_names)


class ElmoReplicaActuator(Actuator[ElmoReplicaActuatorCfg]):
  """Stateful cascaded P/PI actuator with current saturation + anti-windup.

  Not a stateless-law actuator (see Actuator.param_names docstring): the PI
  integrator (`Iv`) is per-(env, target) state carried across steps, so this
  does not participate in automatic fusion and implements `compute`/`update`/
  `reset` directly, following the same pattern as FiniteDifferencePdActuator.
  """

  def __init__(
    self,
    cfg: ElmoReplicaActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.Iv: torch.Tensor | None = None
    self._last_ev: torch.Tensor | None = None
    self._last_u: torch.Tensor | None = None
    self._last_u_sat: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # MuJoCo-side forcerange is a hard backstop matching the drive's peak (if
    # known) or continuous rating -- the real saturation/anti-windup logic
    # lives in compute()/update() below, this is a redundant safety clamp.
    backstop = self.cfg.tau_max_peak or self.cfg.tau_max_continuous
    for target_name in target_names:
      actuator = create_motor_actuator(
        spec,
        target_name,
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
    shape = (data.nworld, len(self._target_names))
    self.Iv = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_ev = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_u = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_u_sat = torch.zeros(shape, device=device, dtype=torch.float)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.Iv is not None

    scale = self.cfg.gear_ratio * RAD_TO_COUNT
    q_ref_count = cmd.position_target * scale
    q_count = cmd.pos * scale
    eq_count = q_ref_count - q_count

    # velocity_target doubles as the velocity feedforward here (rad/s -> counts/s),
    # matching how FiniteDifferencePdActuator's estimated velocity feeds the Kd term.
    # Zero by default (data.joint_vel_target starts at 0), matching the real robot's
    # current FF-disabled state (see RHPS1_gains/README.md "Feedforward and Sampling").
    v_ff_count = cmd.velocity_target * scale
    v_ref_count = self.cfg.Kp_pos * eq_count + v_ff_count

    qdot_count = cmd.vel * scale
    ev = v_ref_count - qdot_count

    u = self.cfg.Kp_vel * ev + self.cfg.Ki_vel * self.Iv
    i_limit = self.cfg.current_limit_continuous
    u_sat = torch.clamp(u, -i_limit, i_limit)
    tau_joint = self.cfg.eta * self.cfg.gear_ratio * self.cfg.torque_constant * u_sat

    # Stash for update(dt): the integrator advance needs dt, which compute() doesn't
    # receive (it runs pre-physics-step; update() runs post-step, see
    # Entity._apply_actuator_controls / Entity.update in mjlab/entity/entity.py).
    self._last_ev = ev
    self._last_u = u
    self._last_u_sat = u_sat

    return tau_joint

  def update(self, dt: float) -> None:
    assert self.Iv is not None
    assert self._last_ev is not None
    assert self._last_u is not None
    assert self._last_u_sat is not None

    # Anti-windup (conditional integration): freeze the integrator while saturated
    # AND the error would push it deeper into saturation. See module docstring --
    # standard assumption, not verified from ELMO docs for this specific drive.
    is_saturating = self._last_u.abs() > self.cfg.current_limit_continuous
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
