from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_height_per_foot_scan(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Per-foot height above the terrain directly beneath that foot, [B, F].

  Same quantity ``split_feet_swing_height.getFootHeightWrtTerrain`` feeds the
  min_foot_height / foot_clearance rewards: the foot site's world z minus the
  mean hit height of that foot's own downward raycast (``<site>_scan``).
  Distinct from ``foot_height`` above, which reads a single whole-body
  TerrainHeightSensor ("foot_height_scan") that the RHPS1 scene does not
  instantiate -- it uses one RayCastSensor per foot instead.

  Intended as a critic-only (privileged) observation: it needs terrain
  raycasts that do not exist on hardware, so it must never reach the actor.
  Motivation is the same as the critic's joint_torques term -- min_foot_height
  charges a one-sided deficit *once per landing*, on the *peak* height reached
  across the whole preceding swing, so the value function has to attribute a
  sparse, trajectory-integrated penalty while never observing the trajectory
  it integrates (Metrics/peak_height_mean sat at ~0.007m through weight ramps
  from -25 to -200). With history, the critic sees the actual height profile
  that produced the peak instead of inferring it from joint angles.
  """
  asset: Entity = env.scene[asset_cfg.name]
  site_names = asset_cfg.site_names
  if site_names is None:
    raise RuntimeError("foot_height_per_foot_scan requires asset_cfg.site_names.")
  if isinstance(site_names, str):
    site_names = (site_names,)

  foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].clone()
  for i, name in enumerate(site_names):
    sensor = env.scene[f"{name}_scan"]
    assert isinstance(sensor, RayCastSensor)
    foot_heights[:, i] -= sensor.data.hit_pos_w[..., 2].mean(dim=-1)
  return foot_heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def pd_action_guidance_target(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  action_term_name: str = "joint_pos",
  max_action_delta: float = 1.0,
) -> torch.Tensor:
  """Raw action that would exactly reproduce this step's clamped PD torque.

  Companion to ``mdp.pd_demand_excess``: instead of scoring the *amount* of
  unclamped-demand overshoot as a reward (routed through the return/advantage,
  where a large-enough weight made the policy timid on the big corrective
  actions needed to catch a stumble -- suppressing exactly the recoveries that
  prevent falls, and a fall's own chaotic demand then fed the same penalty
  back in, a loop that broke two curriculum ramps in a row), this term is
  consumed as a supervised regression target for an auxiliary actor loss
  (``rl_ext.TorqueGuidedPPO``), bypassing the critic/advantage entirely.

  Per joint: demand = kp*(q*-q) + kd*(v*-v) (unclamped); applied = clamp to
  force_limit. The delta between them, converted from torque to action units
  via kp*scale, is exactly the change in this step's raw action that would
  have produced ``applied`` instead of ``demand`` -- zero whenever the joint
  was not saturated, so a fully-legitimate max-effort action that lands
  exactly at the limit is not penalized, only the wasted overshoot beyond it.

  ``max_action_delta`` clips that delta before it reaches the regression
  target. Without it, a mid-fall joint (huge, chaotic q_err) demands an
  equally huge, one-off "correction" that the MSE auxiliary loss squares --
  early in training, when most envs are still falling on every episode, that
  is most of the batch, and it dominated the loss and actively degraded
  balance instead of teaching torque feasibility (observed: fell_down and
  Mean torque_guidance loss climbing together from iteration 0). The cap
  bounds the worst case to a plausible one-step correction regardless of how
  degenerate the underlying state is.
  """
  asset: Entity = env.scene[asset_cfg.name]
  data = asset.data

  demand_full = torch.zeros_like(data.joint_pos)
  limit_full = torch.zeros_like(data.joint_pos)
  stiffness_full = torch.ones_like(data.joint_pos)

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
    demand_full[:, ids] = stiffness * q_err + act.damping * v_err
    limit_full[:, ids] = act.force_limit
    stiffness_full[:, ids] = stiffness

  applied_full = torch.clamp(demand_full, min=-limit_full, max=limit_full)
  dq_equiv = (applied_full - demand_full) / stiffness_full.clamp(min=1e-6)

  action_term = env.action_manager.get_term(action_term_name)
  ids = action_term.target_ids
  raw_action = action_term.raw_action
  scale = action_term.scale
  if not torch.is_tensor(scale):
    scale = torch.full_like(raw_action, float(scale))

  d_action = dq_equiv[:, ids] / scale
  d_action = torch.clamp(d_action, min=-max_action_delta, max=max_action_delta)
  return raw_action + d_action


def gait_phase_obs(
  env: ManagerBasedRlEnv,
  reward_name: str = "gait_phase",
) -> torch.Tensor:
  """[sin(2*pi*phase_left), cos(2*pi*phase_left), sin(2*pi*phase_right),
  cos(2*pi*phase_right)] read off ``mdp.gait_phase_tracking``'s live clock
  (via the registered reward term's own instance -- reward runs before
  observations each step, see ManagerBasedRlEnv.step, so this always reads
  the value that just drove this step's reward).

  sin/cos rather than a raw scalar phase: avoids the 0->1 wraparound
  discontinuity a raw phase value would hand the network. Right block is
  exactly the negation of the left (phase_right = phase_left + 0.5, and
  sin/cos are pi-antiperiodic) -- redundant information, but gives the
  policy a ready per-leg feature instead of making it learn the phase-shift
  relationship itself, and makes the mirror rule a plain block swap (see
  rl_ext.py's _TERM_RULES) with no extra sign math needed.

  Scaled by ``mdp.gait_phase_tracking``'s own ``amplitude`` (0 at zero
  command, ramping to 1 by command_threshold) before being returned, so the
  whole block goes flat to zero as the command nears zero instead of
  freezing at whatever swing/stance encoding the clock happened to stop on.
  A frozen-but-nonzero encoding is otherwise indistinguishable from an
  active gait to the network (see mdp.gait_phase_tracking docstring) and
  was the reason standing episodes kept single-support marching in place
  even though the reward for it was already gated to zero.

  ManagerBasedRlEnv builds observation_manager before reward_manager (see
  load_managers), and observation_manager's own __init__ does a probe
  compute() to size the groups -- so this can run once before
  env.reward_manager exists at all. Fall back to a zero phase/amplitude
  then; the real reward term is live for every actual step after setup
  finishes.
  """
  reward_manager = getattr(env, "reward_manager", None)
  phase = None
  amplitude = None
  if reward_manager is not None:
    func = reward_manager.get_term_cfg(reward_name).func
    phase = func.phase
    amplitude = func.amplitude
  if phase is None:
    phase = torch.zeros(env.num_envs, device=env.device)
  if amplitude is None:
    amplitude = torch.zeros(env.num_envs, device=env.device)
  angle_left = 2.0 * math.pi * phase
  angle_right = 2.0 * math.pi * ((phase + 0.5) % 1.0)
  return amplitude.unsqueeze(-1) * torch.stack(
    [
      torch.sin(angle_left),
      torch.cos(angle_left),
      torch.sin(angle_right),
      torch.cos(angle_right),
    ],
    dim=-1,
  )
