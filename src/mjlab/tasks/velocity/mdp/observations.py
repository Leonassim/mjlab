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
  scan_sensor_names: tuple[str, ...] = ("left_foot_scan", "right_foot_scan"),
) -> torch.Tensor:
  """Per-foot height above the terrain directly beneath that foot, [B, F].

  Same quantity ``split_feet_swing_height.getFootHeightWrtTerrain`` feeds the
  min_foot_height / foot_clearance rewards: the foot site's world z minus the
  mean hit height of that foot's own downward raycast (``<site>_scan``).
  Distinct from ``foot_height`` above, which reads a single whole-body
  TerrainHeightSensor ("foot_height_scan") that the RHPS1 scene does not
  instantiate -- it uses one RayCastSensor per foot instead.

  ``scan_sensor_names`` is an explicit parameter rather than derived from the
  site names at call time, so play.py's --fast sensor pruning can see the
  dependency: it keeps only sensors whose names appear in observation params,
  and a name built at runtime is invisible to it.

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

  if len(scan_sensor_names) != len(site_names):
    raise RuntimeError(
      f"foot_height_per_foot_scan: {len(scan_sensor_names)} scan sensors for "
      f"{len(site_names)} sites"
    )
  foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].clone()
  for i, sensor_name in enumerate(scan_sensor_names):
    sensor = env.scene[sensor_name]
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


def executed_action(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """The policy's previous action as actually executed, in action units.

  Replaces ``mdp.last_action`` for RHPS1. ``last_action`` returns the raw output
  of the network, but the actuator projects any command whose PD demand exceeds
  the effort limit back onto the executable set (see
  ``FiniteDifferencePdActuator.torque_feasibility_ratio``). On the steps where
  that projection bites -- 18% of them early in the 2026-07-29 run -- the raw
  action is a record of intent, not of what happened, so the policy's own
  history misrepresents the dynamics it is trying to model.

  This matters here even though torque saturation itself does not. High-gain
  position control rides its limit as a matter of course, and spending the whole
  torque budget is normal. The defect is that two different commands beyond the
  window produce identical execution: the policy cannot tell them apart, and
  feeding back the intent rather than the execution hides that from it entirely.
  Feeding back the clipped action is the standard fix wherever actions are
  clipped; the projection is just a state-dependent clip.

  Falls back to the raw action for any actuator that does not record an executed
  target (i.e. any actuator without the feasibility projection enabled).
  """
  term = env.action_manager.get_term(action_name)
  raw = env.action_manager.action
  robot: Entity = env.scene[asset_cfg.name]

  q_exec = robot.data.joint_pos.clone()
  found = False
  for act in robot.actuators:
    executed = getattr(act, "_executed_position_target", None)
    if executed is None:
      continue
    q_exec[:, act.target_ids] = executed
    found = True
  if not found:
    return raw

  return (q_exec[:, term.target_ids] - term._offset) / term._scale


def gather_raw_torque_peak(
  robot: Entity, num_joints: int | None = None
) -> torch.Tensor | None:
  """Per-joint peak ``|tau_raw| / effort_limit`` over the last policy step.

  ``tau_raw`` is the PD sum *before* MuJoCo's effort clamp, peak-held across the
  decimation window by FiniteDifferencePdActuator. Returns None if no actuator
  records it, so callers can fail loudly rather than silently observe zeros.

  Normalised by the effort limit on purpose: the per-joint limits span 13 N.m
  (head) to 140 (hip pitch), so raw newton-metres would hand the network ten
  different scales for the same physical quantity. 1.0 is the limit, whatever the
  joint.
  """
  n = num_joints if num_joints is not None else robot.data.joint_pos.shape[1]
  out = torch.zeros(
    (robot.data.joint_pos.shape[0], n),
    device=robot.data.joint_pos.device,
    dtype=torch.float,
  )
  found = False
  for act in robot.actuators:
    peak = getattr(act, "_raw_torque_peak", None)
    if peak is None:
      continue
    out[:, act.target_ids] = peak
    found = True
  return out if found else None


def raw_torque_ratio(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Actor observation: the peak raw-torque ratio of the last policy step.

  The critic already sees ``joint_torques``, but that is *post* clamp: past the
  limit every command reports the same value, so it cannot say how far past. The
  actor saw nothing at all and had to infer saturation from joint_pos/joint_vel.
  This closes that loop, and it is what makes a penalty on the raw torque
  learnable rather than a tax the policy cannot attribute.
  """
  robot: Entity = env.scene[asset_cfg.name]
  peak = gather_raw_torque_peak(robot)
  if peak is None:
    raise RuntimeError(
      "raw_torque_ratio: no actuator records _raw_torque_peak. This observation "
      "requires FiniteDifferencePdActuator; observing zeros would be silent."
    )
  return peak[:, asset_cfg.joint_ids]


def joint_vel_encoder_finite_difference(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  encoder_noise: float = 0.01,
) -> torch.Tensor:
  """Vitesse articulaire telle que le robot reel la fabrique, pas telle que le
  simulateur la connait.

  Sur RHPS1, mc_rtc ne recoit **aucune** vitesse articulaire : verifie de bout
  en bout le 2026-08-06, les drives la mesurent (PDO 0x6069), l'IOB la range
  dans ``state.speed[]`` et ``RobotHardware`` la publie sur son port ``dq``,
  mais ``nocnoid.py`` ne connecte jamais ce port a ``alphaIn``. Le
  ``EncoderObserver`` retombe donc sur ``encoderFiniteDifferences`` et derive
  les positions.

  Ce que l'entrainement faisait jusqu'ici -- verite MuJoCo plus un bruit blanc
  de +/-1.5 rad/s -- a la bonne amplitude par accident, mais pas la bonne
  structure. Une derivee de positions bruitees est **anti-correlee** d'un pas au
  suivant (le meme echantillon de position apparait avec un + puis un -), ce
  qu'une politique peut apprendre a filtrer ; un bruit blanc, non.

  D'ou : on bruite la position AVANT de deriver, dans le terme, parce que le
  bruit d'observation de mjlab s'applique apres la fonction et donnerait une
  derivee propre suivie d'un bruit blanc -- exactement ce qu'on veut quitter.

  ``encoder_noise`` est la demi-largeur du bruit uniforme sur la position, en
  radians, a garder egale a celle du terme ``joint_pos`` (0.01 par defaut).
  L'amplitude qui en sort, 0.01/sqrt(3) * sqrt(2) / 0.005 ~ 1.6 rad/s d'ecart
  type, tombe pres du +/-1.5 blanc qu'elle remplace : le changement porte sur la
  correlation, pas sur le niveau.

  Le biais d'encodeur n'entre pas ici : constant, il disparait dans la
  difference. C'est ``joint_pos`` qui le porte.
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  q = asset.data.joint_pos_biased[:, jnt_ids]
  if encoder_noise > 0.0:
    q = q + (torch.rand_like(q) * 2.0 - 1.0) * encoder_noise

  # Un cache par pas, pas seulement un etat : ce terme est dans le groupe actor
  # ET dans le critic, donc il est appele deux fois par pas. Sans le cache, le
  # deuxieme appel derive q contre lui-meme et rend zero -- un des deux reseaux
  # observerait une vitesse identiquement nulle, en silence. La cle est
  # (common_step_counter, forme), le tirage de bruit est donc partage par les
  # deux groupes, ce qui est correct : c'est le meme capteur.
  key = (int(env.common_step_counter), tuple(q.shape))
  cached = getattr(env, "_encoder_fd_cache", None)
  if cached is not None and cached[0] == key:
    return cached[1]

  prev = getattr(env, "_encoder_fd_prev_q", None)
  if prev is None or prev.shape != q.shape:
    prev = q.clone()

  # Au premier pas d'un episode la difference n'a pas de passe : la mettre a
  # zero plutot que de deriver contre l'etat d'avant la reinitialisation, qui
  # produirait une pointe de vitesse a chaque reset.
  fresh = (env.episode_length_buf == 0).unsqueeze(-1)
  vel = torch.where(fresh, torch.zeros_like(q), (q - prev) / env.step_dt)
  env._encoder_fd_prev_q = q  # type: ignore[attr-defined]
  env._encoder_fd_cache = (key, vel)  # type: ignore[attr-defined]
  return vel


def log_sole_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Hauteur de semelle, mesuree sans passer par un evenement de contact.

  Remplace Metrics/peak_height_mean, qui sous-estimait d'un facteur ~70.
  Celle-la etait publiee par split_feet_swing_height, qui remet son pic a zero
  des le PREMIER coin qui touche alors que le booleen d'atterrissage est vrai
  pour chacun des quatre, sur plusieurs pas successifs : le vrai pic n'etait
  compte qu'une fois et les coins suivants ajoutaient des zeros a la moyenne. Le
  biais etait maximal quand le pied deroule -- le defaut meme qu'on suivait. Le
  2026-08-08 elle annoncait 0.0005 m la ou la mesure directe donnait 0.037.

  Ici : aucun evenement, aucun etat, aucune remise a zero. On lit la hauteur des
  sites de semelle a chaque pas et on publie des quantiles. La mediane dit la
  phase d'appui, le p99 dit la levee. Rien a esquiver.

  Le sol est estime par le p1 de l'echantillon plutot que suppose a zero, pour
  rester juste si le terrain n'est pas plat.
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  floor = torch.quantile(z.flatten(), 0.01)
  rel = z - floor
  log = env.extras.setdefault("log", {})
  log["Metrics/sole_height_p50"] = torch.quantile(rel.flatten(), 0.50)
  log["Metrics/sole_height_p90"] = torch.quantile(rel.flatten(), 0.90)
  log["Metrics/sole_height_p99"] = torch.quantile(rel.flatten(), 0.99)
  log["Metrics/sole_height_max"] = rel.max()
  return rel.max(dim=1).values
