"""Actions ignored for a while at the start of a standing-start episode.

mc_mujoco holds the robot at q0 for 3 s before arming, so the policy takes over a
SETTLED robot. Training let it act from the first step, on a state written the
same instant. Measured 2026-09-19 on model_2500: 2.07 m in 12 s acting
immediately, 0.05 m after a 3 s settle -- and 0.05 m in mc_mujoco. The start
transient was the trigger, not the command.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


def stance_action(env, term) -> torch.Tensor:
  """Action that holds the start posture: (q_start - q0) / scale, zero without an offset."""
  off = getattr(env, "_hold_offset_q", None)
  if off is None:
    return torch.zeros(env.num_envs, term.action_dim, device=env.device)
  a = off[:, term.target_ids] / term._scale
  vel = getattr(term, "_vel", None)
  return a if vel is None else a * ~vel  # velocity joints hold with zero velocity


class HoldableJointPositionAction(JointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._cyl = None
    from mjlab.asset_zoo.robots.RHPS1.rhps1_cylinders import cylinder_limit_tensors, cylinder_limits_enabled

    if cylinder_limits_enabled():
      lim = self._entity.data.joint_pos_limits[0, self._target_ids]
      self._cyl = cylinder_limit_tensors(list(self.target_names), lim, self.device)
    from mjlab.tasks.tracking.config.rhps1 import vision_map

    names = list(self.target_names)
    self._fixed = [(names.index("HEAD_P"), vision_map.HEAD_P)] if vision_map.HEAD_HOLD and "HEAD_P" in names else []

  def process_actions(self, actions: torch.Tensor) -> None:
    hold = getattr(self._env, "_action_hold_steps", None)
    if hold is not None:
      held = self._env.episode_length_buf < hold
      if bool(held.any()):
        actions = torch.where(held[:, None], stance_action(self._env, self), actions)
    super().process_actions(actions)
    if self._cyl is not None:
      # Same target projection as rl_controller (cylinder_limits/project_targets): part of the deployed plant.
      from mjlab.asset_zoo.robots.RHPS1.rhps1_cylinders import project_cylinder_limits

      self._processed_actions = project_cylinder_limits(self._processed_actions, *self._cyl)
    for i, v in self._fixed:  # joint held by the controller, whatever the policy outputs
      self._processed_actions[:, i] = v


@dataclass(kw_only=True)
class HoldableJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env) -> HoldableJointPositionAction:
    return HoldableJointPositionAction(self, env)


class HoldableJointVelocityAction(HoldableJointPositionAction):
  """RHPS1_ACTION_MODE=vel: the joints whose name contains a vel_tokens entry take a velocity, the target is its
  free-running integral, as rl_controller's velocity_action (utils.cpp): integrated every physics step, kept within
  max(effort/kp, 2|v|/sqrt(K)) of the measured (motor side, encoder) position, then the joint range and the cylinder
  projection. Other joints stay position actions. Noise moves the target by v*dt per step instead of a full jump."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    names = list(self.target_names)
    vel = [any(t in n for t in cfg.vel_tokens) for n in names]
    self._vel = torch.tensor(vel, dtype=torch.bool, device=self.device)
    self._vcols = self._vel.nonzero().flatten()
    vnames = [names[i] for i in self._vcols.tolist()]
    self._vscale = torch.tensor([cfg.vel_scale[n] for n in vnames], device=self.device)
    self._dev0 = torch.tensor([cfg.dev_floor[n] for n in vnames], device=self.device)
    vs = vnames
    self._vmax = cfg.vel_max
    self._sqrt_k = cfg.posture_k**0.5 if cfg.posture_k > 0 else 0.0
    self._dt = env.cfg.sim.mujoco.timestep
    lim = self._entity.data.joint_pos_limits[0, self._target_ids[self._vcols]]
    self._lo, self._hi = lim[:, 0].clone(), lim[:, 1].clone()
    self._v = torch.zeros(self.num_envs, len(vs), device=self.device)
    self._int = torch.zeros(self.num_envs, len(vs), device=self.device)
    self._init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self._target = torch.zeros(self.num_envs, self.action_dim, device=self.device)

  def _measured(self) -> torch.Tensor:
    ids = self._target_ids[self._vcols]
    q = self._entity.data.joint_pos[:, ids] + self._entity.data.encoder_bias[:, ids]
    d = getattr(self._entity, "_rhps1_deflection", None)
    return q if d is None else q + d[:, ids]

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    self._init[slice(None) if env_ids is None else env_ids] = True

  def process_actions(self, actions: torch.Tensor) -> None:
    hold = getattr(self._env, "_action_hold_steps", None)
    a = actions[:, self._vcols]
    if hold is not None:
      a = torch.where((self._env.episode_length_buf < hold)[:, None], torch.zeros_like(a), a)
    self._v = (a * self._vscale).clamp(-self._vmax, self._vmax)
    if bool(self._init.any()):  # first step of an episode: start from the reset posture
      self._int = torch.where(self._init[:, None], self._measured(), self._int)
      self._init[:] = False
    super().process_actions(actions)  # position joints, hold, head; leg columns overwritten in apply_actions
    self._target = self._processed_actions.clone()

  def apply_actions(self) -> None:
    q = self._measured()
    dev = self._dev0
    if self._sqrt_k > 0:
      dev = torch.maximum(dev, 2.0 * self._v.abs() / self._sqrt_k)
    i = torch.minimum(torch.maximum(self._int + self._v * self._dt, q - dev), q + dev)
    i = torch.minimum(torch.maximum(i, self._lo), self._hi)
    t = self._target.clone()
    t[:, self._vcols] = i
    for j, v in self._fixed:
      t[:, j] = v
    if self._cyl is not None:
      from mjlab.asset_zoo.robots.RHPS1.rhps1_cylinders import project_cylinder_limits

      t = project_cylinder_limits(t, *self._cyl)
    self._int = t[:, self._vcols]
    self._processed_actions = t
    super(HoldableJointPositionAction, self).apply_actions()


# mc_mujoco slot 18 (2026-11-25, forward / left / yaw, 20 ms means): p99 |qd| per joint, floor 0.3 rad/s. The
# velocity action scale (1 unit = this speed) and the stairs_vrate normalisation.
SLOT18_P99_VEL = {
  "L_CROTCH_Y": 0.4, "L_CROTCH_R": 0.6, "L_CROTCH_P": 1.1, "L_KNEE_P": 2.2, "L_ANKLE_R": 0.5, "L_ANKLE_P": 0.5,
  "R_CROTCH_Y": 0.4, "R_CROTCH_R": 0.6, "R_CROTCH_P": 1.1, "R_KNEE_P": 2.2, "R_ANKLE_R": 0.5, "R_ANKLE_P": 0.5,
  "CHEST_Y": 1.0, "CHEST_P": 0.3, "HEAD_Y": 0.4, "HEAD_P": 0.4,
  "L_SHOULDER_P": 0.3, "L_SHOULDER_R": 0.3, "L_SHOULDER_Y": 0.3, "L_ELBOW_P": 0.3, "L_ELBOW_Y": 0.3,
  "L_WRIST_R": 0.5, "L_WRIST_Y": 0.4,
  "R_SHOULDER_P": 0.3, "R_SHOULDER_R": 0.3, "R_SHOULDER_Y": 0.3, "R_ELBOW_P": 0.3, "R_ELBOW_Y": 0.3,
  "R_WRIST_R": 0.5, "R_WRIST_Y": 0.4,
}


@dataclass(kw_only=True)
class HoldableJointVelocityActionCfg(HoldableJointPositionActionCfg):
  vel_tokens: tuple[str, ...] = ("",)  # every joint; ("CROTCH", "KNEE", "ANKLE") for the legs only
  vel_scale: dict[str, float] = None  # rad/s per action unit, per joint name
  dev_floor: dict[str, float] = None  # effort / kp per joint name
  vel_max: float = 3.0
  posture_k: float = 1600.0

  def build(self, env) -> HoldableJointVelocityAction:
    return HoldableJointVelocityAction(self, env)


def held_last_action(env, action_name: str | None = None) -> torch.Tensor:
  """Last action as the controller presents it: zero while the hold zeroed it.

  The raw output during the hold is never applied; rl_controller arms with the
  stance's own action (~0 for the legs), so a policy trained on its raw outputs
  jerked at arming (2026-10-23: 41500 hit 5.6 rad/s, 80/80 falls).
  """
  a = env.action_manager.action
  hold = getattr(env, "_action_hold_steps", None)
  if hold is None:
    return a
  # The action applied at step t was held iff t < hold, i.e. buf (= t + 1) <= hold.
  held = env.episode_length_buf <= hold
  term = env.action_manager.get_term(action_name or "joint_pos")
  return torch.where(held[:, None], stance_action(env, term), a)



def _not_held(env, out: torch.Tensor) -> torch.Tensor:
  hold = getattr(env, "_action_hold_steps", None)
  return out if hold is None else out & ~(env.episode_length_buf <= hold)


def held_bad_anchor_pos_z_only(env, **kwargs) -> torch.Tensor:
  """Tracking terminations that cannot fire while the hold ignores the policy."""
  from mjlab.tasks.tracking import mdp

  return _not_held(env, mdp.bad_anchor_pos_z_only(env, **kwargs))


def held_bad_anchor_ori(env, **kwargs) -> torch.Tensor:
  from mjlab.tasks.tracking import mdp

  return _not_held(env, mdp.bad_anchor_ori(env, **kwargs))


def held_bad_motion_body_pos_z_only(env, **kwargs) -> torch.Tensor:
  from mjlab.tasks.tracking import mdp

  return _not_held(env, mdp.bad_motion_body_pos_z_only(env, **kwargs))
