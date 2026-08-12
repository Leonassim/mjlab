"""PD actuator with desired velocity estimated from position target changes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque

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

  posture_task_stiffness: float | None = None
  """Raideur de la PostureTask mc_rtc a reproduire, ``None`` pour desactiver.

  Sur le robot, la sortie de la politique ne va PAS directement au PD : elle
  traverse d'abord la ``PostureTask`` du QP, un second ordre en espace
  articulaire de pulsation ``sqrt(K)`` et d'amortissement ``2*sqrt(K)``
  (mc_rtc calcule l'amortissement automatiquement). A K=1600, la valeur retenue
  pour le robot ET pour mc_mujoco, cela fait 40 rad/s, soit environ 25 ms de
  constante de temps -- cinq pas de politique de retard que l'entrainement ne
  voyait pas.

  Ce filtre place le meme second ordre en amont du PD :

      qdd_f = K*(q_cmd - q_f) - 2*sqrt(K)*qd_f

  Le PD suit ensuite ``q_f`` au lieu de ``q_cmd``. Tout le reste de cet
  actionneur -- difference finie, EMA de vitesse, projection de faisabilite --
  travaille sur la sortie du filtre, comme sur le robot ou ces etages sont bien
  en aval du QP.
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

  torque_feasibility_ratio: float | None = None
  """Cap on ``|kp*(q*-q) + kd*(qd*-qd)| / effort_limit``, enforced by projecting
  the position target. ``None`` disables the projection.

  Why this exists: without it the PD demand is clamped *silently* by the
  effort limit, so every action that asks for more than the limit produces the
  exact same torque, the exact same dynamics and the exact same reward. PPO is
  model-free -- no gradient crosses that clamp -- so the whole saturated region
  is one flat plateau in the return, the policy mean random-walks inside it,
  and the exploration noise is invisible there too (entropy becomes free, std
  runs away, the executed torque degenerates into bang-bang). Projecting the
  target instead keeps every commanded action executable, which is what makes
  the reward landscape informative near the limit and what makes a
  position-controlled deployment faithful to training.

  Set this together with the leg ``action_scale``: the projection window is
  ``ratio * effort_limit / stiffness``, which is exactly ``ratio`` raw action
  units when the scale equals the actuator's own ``effort_limit / stiffness``
  (``_LEG_SCALE_MULTIPLIER = 1.0``). At a 7x leg scale the same window is 1/7
  of an action unit, i.e. nearly every action lands on the projection and the
  plateau comes straight back -- worse than no projection at all.

  Use ``1.0``, always, and do not ramp it. At ``1.0`` this projection is not a
  new constraint on the physics at all -- it delivers *exactly* the torque
  MuJoCo's effort clamp already delivered:

      tau(q*) = kp*(q* - q) + kd*(qd* - qd)  is affine and increasing in q*,
      so clamping tau to [-e, e] and projecting q* onto tau's preimage of
      [-e, e] are the same operation. Above the window both give +e, below it
      both give -e, inside it neither changes anything.

  So it cannot strangle a gait, cap a joint velocity, or alter a reward: same
  applied torque, same dynamics, same return. What it changes is the *record*:
  the target that leaves this actuator is now one a plain PD with no clamp
  downstream would execute identically. That is the entire point -- mc_rtc in
  position/QP mode has no torque clamp, so a policy whose commanded target only
  works because something clips it does not transfer, while one whose target is
  already inside the window transfers exactly.

  Any ratio above 1.0 gives the identical physics too (MuJoCo clamps the
  residual), and buys nothing: it just re-admits the un-executable commands
  this exists to remove. There is no trade-off to tune, hence no curriculum.

  Note this projection is local to ``compute``: ``data.joint_pos_target`` keeps
  the raw, un-projected target, so ``pd_demand_excess`` and
  ``Metrics/pd_demand_ratio_*`` still measure how far the policy's *command*
  is from feasible. That is deliberate -- it is the gradient that moves the raw
  action into the window, which the projection alone cannot supply.
  """

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
    # Etat du second ordre reproduisant la PostureTask. Voir
    # posture_task_stiffness. Le pas de sous-etape n'est connu que dans
    # update(dt) ; tant qu'il ne l'est pas, le filtre laisse passer.
    self._posture_q: torch.Tensor | None = None
    self._posture_qd: torch.Tensor | None = None
    # Raideur par environnement plutot que scalaire de cfg : elle est
    # randomisable (voir randomize_posture_task_stiffness). Seedee depuis la
    # cfg a l'initialisation ; reste uniforme si personne ne la randomise.
    self.posture_stiffness: torch.Tensor | None = None
    self.default_posture_stiffness: torch.Tensor | None = None
    self._substep_dt: float | None = None
    self._desired_velocity_target: torch.Tensor | None = None
    self._elapsed_since_target_update: torch.Tensor | None = None
    self._initialized: torch.Tensor | None = None
    self._q_lower: torch.Tensor | None = None
    self._q_upper: torch.Tensor | None = None
    self._v_max: torch.Tensor | None = None
    # Curriculum progress [0, 1]: 0 = no damper, 1 = full mc_rtc QP constraints.
    self.velocity_damper_progress: float = 0.0
    # Curriculum progress (0, 1]: the effective ratio is
    # ``torque_feasibility_ratio / progress``, so 1 = the configured ratio and
    # values below 1 loosen it (0 = disabled). Unlike the damper this defaults
    # to fully active: it is a structural constraint on the action space, not a
    # penalty routed through the advantage, so there is no timidity loop to ramp
    # around -- the entire point is that exploration never leaves the feasible
    # set. Drop it below 1 for the early iterations only if a gait fails to form
    # at all.
    self.torque_feasibility_progress: float = 1.0
    # The position target as actually executed, recorded on the substep where a
    # new command arrives. Fed back to the policy in place of its raw action:
    # what a policy observes about its own past has to be what happened, not
    # what it asked for, or the dynamics look inconsistent on exactly the steps
    # where the projection bit. Same class of defect as the EMA hidden state
    # this actuator no longer carries.
    self._executed_position_target: torch.Tensor | None = None
    # Peak |tau_raw| / effort_limit over the physics substeps of one policy step,
    # where tau_raw is the PD sum BEFORE MuJoCo's effort clamp. Two reasons this
    # is the pre-clamp value and not data.actuator_force:
    #   - post-clamp, every command past the limit reports the same torque, so
    #     the quantity carries no information about how far past it went;
    #   - the peak is what sizes the hardware, and it lives inside the decimation
    #     window, invisible to anything sampled at the policy rate.
    # Fed to the actor as an observation and charged by a reward term, which is
    # what gives PPO a gradient across the clamp -- the job the target projection
    # was doing structurally.
    self._raw_torque_peak: torch.Tensor | None = None
    # Same peak-hold, on each half of the PD sum separately: kp*(q*-q) and
    # kd*(qd*-qdot). Diagnostic only, and note these are peak-held independently,
    # so they do not add up to _raw_torque_peak -- each answers "how big does this
    # half get", not "what was the other half worth at the peak instant".
    #
    # Why it matters: a ratio of 75 on CROTCH_Y (35 N.m, kp 20000, kd 400, action
    # scale 7*e/kp) needs either q*-q = 0.131 rad, i.e. 10.7 raw action units,
    # which the policy never emits -- or qd*-qdot = 6.6 rad/s, which is ordinary
    # for a swinging leg. If the kd half dominates, a penalty on the total is
    # really asking the policy to move its joints more slowly, which fights the
    # gait, rather than to stop commanding impossible targets.
    self._raw_torque_peak_kp: torch.Tensor | None = None
    self._raw_torque_peak_kd: torch.Tensor | None = None

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
    # Allocated here, not lazily in compute(). compute() runs under
    # torch.inference_mode() during rollouts, so a buffer created there is an
    # inference tensor and reset() may not write to it from outside that mode --
    # training happens to survive (its resets are called from inside the rollout)
    # but play and any offline evaluation raise. Same pattern as the buffers
    # below, which is what it should have been from the start.
    self._executed_position_target = torch.zeros(
      shape, device=device, dtype=torch.float
    )
    self._raw_torque_peak = torch.zeros(shape, device=device, dtype=torch.float)
    self._raw_torque_peak_kp = torch.zeros(shape, device=device, dtype=torch.float)
    self._raw_torque_peak_kd = torch.zeros(shape, device=device, dtype=torch.float)
    self._last_position_target = torch.zeros(shape, device=device, dtype=torch.float)
    self._filtered_position_target = torch.zeros(
      shape, device=device, dtype=torch.float
    )
    self._posture_q = torch.zeros(shape, device=device, dtype=torch.float)
    self._posture_qd = torch.zeros(shape, device=device, dtype=torch.float)
    if self.cfg.posture_task_stiffness is not None:
      self.posture_stiffness = torch.full(
        shape, float(self.cfg.posture_task_stiffness), device=device,
        dtype=torch.float)
      self.default_posture_stiffness = self.posture_stiffness.clone()
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
      if self._posture_q is not None:
        # Seeder sur la cible, pas sur zero : sinon le filtre demarre a la
        # position nulle et envoie le robot chercher q=0 au premier pas.
        self._posture_q = torch.where(uninitialized, cmd.position_target,
                                      self._posture_q)
        assert self._posture_qd is not None
        self._posture_qd = torch.where(uninitialized,
                                       torch.zeros_like(self._posture_qd),
                                       self._posture_qd)

    # PostureTask du QP, reproduite en amont de tout le reste : sur le robot le
    # QP est bien avant la difference finie et la projection.
    posture_k = self.posture_stiffness
    if posture_k is not None and self._substep_dt is not None:
      assert self._posture_q is not None and self._posture_qd is not None
      dt = self._substep_dt
      # Amortissement critique 2*sqrt(K), comme mc_rtc le calcule lui-meme --
      # il suit donc K quand celui-ci est randomise, au lieu de rester fige.
      acc = posture_k * (cmd.position_target - self._posture_q) \
          - 2.0 * torch.sqrt(posture_k) * self._posture_qd
      # Semi-implicite : vitesse d'abord, puis position avec la vitesse a jour.
      # Plus stable que l'explicite pur pour le meme cout, ce qui compte ici
      # puisque sqrt(K)*dt vaut 0.1 a K=1600 et dt=0.0025.
      self._posture_qd = self._posture_qd + acc * dt
      self._posture_q = self._posture_q + self._posture_qd * dt
      cmd = replace(cmd, position_target=self._posture_q)

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

    # Captured BEFORE the damper and the feasibility projection. Everything below
    # this line bends the target back inside the actuator's budget, so a torque
    # computed from the result is capped at exactly the effort limit by
    # construction -- measured, it reads 1.0000 on every joint, which looks like a
    # perfectly saturated robot and is really just the projection reporting itself.
    # The quantity worth observing and charging is the one the policy *asked* for.
    pre_projection_target = filtered_position_target

    # Damper first, feasibility last. The order matters for a specific reason:
    # applied last, the feasibility projection makes the delivered torque
    # *provably identical* to what MuJoCo's effort clamp would have produced
    # from the un-projected target -- clamping tau and projecting q* onto the
    # preimage of that clamp are the same operation, since tau is affine and
    # increasing in q*. Anything applied after it would break that identity.
    filtered_position_target = self._apply_velocity_damper(
      filtered_position_target, cmd.pos, cmd.vel
    )
    filtered_position_target = self._apply_torque_feasibility(
      filtered_position_target, cmd.pos, cmd.vel
    )

    # Record on the substep that carries a fresh command. Later substeps
    # re-project against evolved state, so they answer a different question --
    # this one is "what did the decision I just made actually become".
    self._executed_position_target = torch.where(
      changed | uninitialized, filtered_position_target, self._executed_position_target
    )

    pd_cmd = ActuatorCmd(
      position_target=filtered_position_target,
      velocity_target=self._desired_velocity_target,
      effort_target=cmd.effort_target,
      pos=cmd.pos,
      vel=cmd.vel,
    )

    # Peak-hold over the decimation window. `changed` marks the substep carrying a
    # fresh command, i.e. the start of a policy step, so the peak resets there and
    # accumulates over the substeps that follow. Same boundary the executed target
    # is recorded on, so the two always describe the same decision.
    assert self.stiffness is not None
    assert self.damping is not None
    assert self.force_limit is not None
    assert self._raw_torque_peak is not None
    raw_cmd = ActuatorCmd(
      position_target=pre_projection_target,
      velocity_target=self._desired_velocity_target,
      effort_target=cmd.effort_target,
      pos=cmd.pos,
      vel=cmd.vel,
    )
    raw_ratio = torch.abs(pd_torque(self.stiffness, self.damping, raw_cmd)) / torch.clamp(
      self.force_limit, min=1e-6
    )
    start = changed | uninitialized
    self._raw_torque_peak = torch.where(
      start, raw_ratio, torch.maximum(self._raw_torque_peak, raw_ratio)
    )

    assert self._raw_torque_peak_kp is not None
    assert self._raw_torque_peak_kd is not None
    inv_lim = 1.0 / torch.clamp(self.force_limit, min=1e-6)
    kp_ratio = torch.abs(self.stiffness * (pre_projection_target - cmd.pos)) * inv_lim
    kd_ratio = (
      torch.abs(self.damping * (self._desired_velocity_target - cmd.vel)) * inv_lim
    )
    self._raw_torque_peak_kp = torch.where(
      start, kp_ratio, torch.maximum(self._raw_torque_peak_kp, kp_ratio)
    )
    self._raw_torque_peak_kd = torch.where(
      start, kd_ratio, torch.maximum(self._raw_torque_peak_kd, kd_ratio)
    )

    return super().compute(pd_cmd)

  def _apply_torque_feasibility(
    self,
    q_target: torch.Tensor,
    q: torch.Tensor,
    qdot: torch.Tensor,
  ) -> torch.Tensor:
    """Project the position target so the PD demand stays inside the budget.

    The demand ``tau = kp*(q* - q) + kd*(qd* - qd)`` is affine in ``q*``, so
    ``|tau| <= budget`` is a closed interval on ``q*``:

        q* in q + [(-budget - kd*v_err) / kp, (budget - kd*v_err) / kp]

    The interval is always non-empty (its width is ``2*budget/kp > 0``); the
    velocity term only shifts it. When the velocity error alone would already
    blow the budget the interval sits entirely on one side of ``q``, i.e. the
    projection commands the joint *back* -- which is exactly what the effort
    clamp was doing implicitly, except now the policy can see it in the return.

    ``qd*`` is the finite-difference estimate computed just above, so this runs
    after velocity estimation and reuses it rather than re-deriving it.
    """
    ratio = self.cfg.torque_feasibility_ratio
    p = self.torque_feasibility_progress
    if ratio is None or p <= 0.0:
      return q_target
    assert self.stiffness is not None
    assert self.damping is not None
    assert self.force_limit is not None
    assert self._desired_velocity_target is not None

    # progress < 1 loosens the cap (progress -> 0 gives an unbounded budget).
    budget = (ratio / p) * self.force_limit
    v_term = self.damping * (self._desired_velocity_target - qdot)
    kp = torch.clamp(self.stiffness, min=1e-6)
    q_lo = q + (-budget - v_term) / kp
    q_hi = q + (budget - v_term) / kp
    return torch.clamp(q_target, q_lo, q_hi)

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
    # Seul endroit ou le pas de sous-etape est connu ; compute() en a besoin
    # pour integrer le filtre de PostureTask.
    self._substep_dt = dt

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    assert self._last_position_target is not None
    assert self._filtered_position_target is not None
    assert self._desired_velocity_target is not None
    assert self._elapsed_since_target_update is not None
    assert self._initialized is not None
    if self._executed_position_target is not None:
      self._executed_position_target[env_ids] = 0.0
    # L'etat du filtre de PostureTask doit repartir de zero avec _initialized,
    # sinon le nouvel episode herite de la posture du precedent et le filtre
    # tire le robot vers elle pendant ses 25 ms de constante de temps.
    for _buf in (self._posture_q, self._posture_qd):
      if _buf is not None:
        _buf[env_ids] = 0.0
    for _buf in (
      self._raw_torque_peak,
      self._raw_torque_peak_kp,
      self._raw_torque_peak_kd,
    ):
      if _buf is not None:
        _buf[env_ids] = 0.0
    self._last_position_target[env_ids] = 0.0
    self._filtered_position_target[env_ids] = 0.0
    self._desired_velocity_target[env_ids] = 0.0
    self._elapsed_since_target_update[env_ids] = 0.0
    self._initialized[env_ids] = False
