"""RHPS1 velocity environment configurations."""

from mjlab.asset_zoo.robots import RHPS1_ACTION_SCALE, get_rhps1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def rhps1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create RHPS1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.timestep = 0.0025  # 400 Hz physics; step_dt = 5 ms (deployment: 250 Hz)
  cfg.decimation = 2
  cfg.sim.mujoco.iterations = 15
  cfg.sim.mujoco.ls_iterations = 30

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 64

  cfg.scene.entities = {"robot": get_rhps1_robot_cfg()}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(L_ANKLE_P_LINK|R_ANKLE_P_LINK)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  feet_ground_split_cfg = ContactSensorCfg(
    name="feet_ground_contact_split",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left_foot[1-4]_collision|right_foot[1-4]_collision)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  feet_mesh_cfg = ContactSensorCfg(
    name="feet_mesh_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left_foot_collision|right_foot_collision)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Force-based (not proximity) so the forceless leg-gap contacts below don't
  # register as self-collisions.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="BODY", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="BODY", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=2,
  )
  # Mirrors the deployment QP's minimalSelfCollisions pairs; thresholds live
  # in the matching leg_proximity_cost reward terms below.
  def _proximity_sensor(name: str, primary: str, secondary: str) -> ContactSensorCfg:
    return ContactSensorCfg(
      name=name,
      primary=ContactMatch(mode="geom", pattern=primary, entity="robot"),
      secondary=ContactMatch(mode="geom", pattern=secondary, entity="robot"),
      fields=("found", "dist"),
      reduce="mindist",
      num_slots=1,
    )

  leg_proximity_cfg = _proximity_sensor(
    "leg_proximity",
    r"^rhps1_collision_L_(CROTCH_P|KNEE_P|ANKLE_R)_LINK$",
    r"^rhps1_collision_R_(CROTCH_P|KNEE_P|ANKLE_R)_LINK$",
  )
  # Wider threshold: mc_rtc knee hulls are ~1.5cm fatter than the mujoco mesh.
  knee_proximity_cfg = _proximity_sensor(
    "knee_proximity",
    r"^rhps1_collision_L_KNEE_P_LINK$",
    r"^rhps1_collision_R_KNEE_P_LINK$",
  )
  arm_torso_proximity_cfg = _proximity_sensor(
    "arm_torso_proximity",
    r"^rhps1_collision_[LR]_(ELBOW_Y|WRIST_Y)_LINK$",
    r"^rhps1_collision_(CHEST_P_LINK|BODY)$",
  )
  shoulder_chest_proximity_cfg = _proximity_sensor(
    "shoulder_chest_proximity",
    r"^rhps1_collision_[LR]_SHOULDER_Y_LINK$",
    r"^rhps1_collision_CHEST_P_LINK$",
  )
  shoulder_body_proximity_cfg = _proximity_sensor(
    "shoulder_body_proximity",
    r"^rhps1_collision_[LR]_SHOULDER_Y_LINK$",
    r"^rhps1_collision_BODY$",
  )
  wrist_thigh_proximity_cfg = _proximity_sensor(
    "wrist_thigh_proximity",
    r"^rhps1_collision_[LR]_WRIST_Y_LINK$",
    r"^rhps1_collision_[LR]_CROTCH_P_LINK$",
  )
  pattern_cfg = GridPatternCfg(
    size=(0.2, 0.2),
    resolution=0.1,
    direction=(0.0, 0.0, -1.0),
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="BODY", entity="robot"),
    pattern=pattern_cfg,
    ray_alignment="yaw",
    max_distance=3.0,
    exclude_parent_body=True,
    include_geom_groups=(0, 1),
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.0, 1.0, 0.0, 0.9),
      miss_color=(1.0, 0.0, 0.0, 0.5),
      show_rays=False,
      show_normals=True,
    ),
  )
  left_foot_raycast_cfg = RayCastSensorCfg(
    name="left_foot_scan",
    frame=ObjRef(type="body", name="L_ANKLE_P_LINK", entity="robot"),
    pattern=pattern_cfg,
    ray_alignment="yaw",
    max_distance=3.0,
    exclude_parent_body=True,
    include_geom_groups=(0, 1),
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.0, 1.0, 0.0, 0.9),
      miss_color=(1.0, 0.0, 0.0, 0.5),
      show_rays=False,
      show_normals=True,
    ),
  )
  right_foot_raycast_cfg = RayCastSensorCfg(
    name="right_foot_scan",
    frame=ObjRef(type="body", name="R_ANKLE_P_LINK", entity="robot"),
    pattern=pattern_cfg,
    ray_alignment="yaw",
    max_distance=3.0,
    exclude_parent_body=True,
    include_geom_groups=(0, 1),
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.0, 1.0, 0.0, 0.9),
      miss_color=(1.0, 0.0, 0.0, 0.5),
      show_rays=False,
      show_normals=True,
    ),
  )
  cfg.scene.sensors = (
    feet_ground_cfg,
    feet_ground_split_cfg,
    feet_mesh_cfg,
    self_collision_cfg,
    leg_proximity_cfg,
    knee_proximity_cfg,
    arm_torso_proximity_cfg,
    shoulder_chest_proximity_cfg,
    shoulder_body_proximity_cfg,
    wrist_thigh_proximity_cfg,
    raycast_cfg,
    left_foot_raycast_cfg,
    right_foot_raycast_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RHPS1_ACTION_SCALE

  actor_group_name = "policy" if "policy" in cfg.observations else "actor"
  history_len = 5
  # Defined up here (not at its first reward use further down) because the
  # critic's foot_height_scan observation below needs it too.
  site_names = ("left_foot", "right_foot")

  old_terms = cfg.observations[actor_group_name].terms
  old_terms.pop("phase", None)
  old_terms.pop("height_scan", None)
  old_terms.pop("base_lin_vel", None)
  base_ang_vel_term = old_terms.get("base_ang_vel")
  if base_ang_vel_term is not None:
    base_ang_vel_term.func = mdp.base_ang_vel
    base_ang_vel_term.params = {}
    base_ang_vel_term.noise = Unoise(n_min=-0.3, n_max=0.3)
  proj_grav_term = old_terms.get("projected_gravity")
  if proj_grav_term is not None:
    proj_grav_term.func = mdp.projected_gravity_biased
    proj_grav_term.params = {}
    proj_grav_term.noise = Unoise(n_min=-0.1, n_max=0.1)
  for term in old_terms.values():
    if getattr(term, "history_length", None) is not None and term.history_length > 1:
      term.history_length = history_len
  if "command" in old_terms:
    old_terms["command"].history_length = history_len
    old_terms["command"].flatten_history_dim = True
  new_terms = {
    "base_lin_vel": ObservationTermCfg(
      # _biased : biais constant par episode, tire par l'evenement
      # MuJoCo ground truth in training, a drifting observer on the robot.
      func=mdp.base_lin_vel_biased,
      noise=Unoise(n_min=-0.1, n_max=0.1),
      history_length=history_len,
      flatten_history_dim=True,
    )
  }
  new_terms.update(old_terms)
  # reward_name explicit, not defaulted: play.py --fast scans params to decide
  # which rewards to keep, and the clock lives in the gait_phase reward.
  new_terms["gait_phase"] = ObservationTermCfg(
    func=mdp.gait_phase_obs, params={"reward_name": "gait_phase"}
  )
  # Executed, not requested: the actuator projects any target whose PD demand
  # exceeds the effort limit, so two different actions can execute identically.
  # History 5 because the velocity-target EMA (alpha 0.8) is hidden state.
  # The C++ controller must feed back the same quantity, see utils.cpp.
  if "actions" in new_terms:
    new_terms["actions"] = ObservationTermCfg(
      func=mdp.executed_action, history_length=history_len, flatten_history_dim=True
    )
  if "gait_phase" in new_terms:
    new_terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase_obs,
      params={"reward_name": "gait_phase"},
      history_length=history_len,
      flatten_history_dim=True,
    )
  # Post-clamp joint_torques cannot say how far past the limit a command went.
  # History 10, local to this block: saturation builds over tens of ms.
  # Needs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True or the run OOMs.
  new_terms["raw_torque"] = ObservationTermCfg(
    func=mdp.raw_torque_ratio,
    history_length=10,
    flatten_history_dim=True,
  )
  cfg.observations[actor_group_name].terms = new_terms

  if "actor_history" in cfg.observations:
    ah_terms = cfg.observations["actor_history"].terms
    ah_terms.pop("phase", None)
    ah_terms.pop("height_scan", None)
    for term in ah_terms.values():
      if getattr(term, "history_length", None) is not None and term.history_length > 1:
        term.history_length = history_len
    if "command" in ah_terms:
      ah_terms["command"].history_length = history_len
      ah_terms["command"].flatten_history_dim = True
    ah_new = {
      "base_lin_vel": ObservationTermCfg(
        func=mdp.base_lin_vel,
        history_length=history_len,
        flatten_history_dim=True,
      )
    }
    ah_new.update(ah_terms)
    ah_new["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase_obs, params={"reward_name": "gait_phase"}
    )
    # Inert: there is no actor_history group in this configuration (the actor
    # carries its own per-term history_length instead). Kept for the variants
    # that do define one; they must agree with the actor above.
    if "actions" in ah_new:
      ah_new["actions"] = ObservationTermCfg(func=mdp.last_action)
    cfg.observations["actor_history"].terms = ah_new

  if "critic" in cfg.observations:
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
      func=mdp.base_lin_vel
    )
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
      func=mdp.base_ang_vel
    )
    # Critic-only (privileged): actual per-joint applied torque, post actuator
    # clamp. Lets the value function see saturation state directly instead of
    # having to infer it from joint_pos/joint_vel, sharpening credit
    # assignment for the torque-related reward terms below. History (same
    # length as base_lin_vel/command in the actor group) so the critic sees
    # a recent torque trajectory, not just an instantaneous snapshot --
    # sustained saturation vs. a one-step spike are very different signals
    # for value estimation.
    cfg.observations["critic"].terms["joint_torques"] = ObservationTermCfg(
      func=mdp.joint_torques,
      history_length=history_len,
      flatten_history_dim=True,
    )
    cfg.observations["critic"].terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase_obs, params={"reward_name": "gait_phase"}
    )
    # Critic-only (privileged, needs terrain raycasts absent on hardware):
    # terrain-relative height of each foot, with history. min_foot_height
    # charges once per landing on the peak height of the whole preceding
    # swing, so without this the value function attributes a sparse,
    # trajectory-integrated penalty while never seeing the trajectory --
    # plausibly part of why peak_height_mean stayed ~0.007m across weight
    # ramps from -25 all the way to -200. Named distinctly from the base
    # config's "foot_height" term (a whole-body TerrainHeightSensor the
    # RHPS1 scene does not instantiate, popped just below).
    cfg.observations["critic"].terms["foot_height_scan"] = ObservationTermCfg(
      func=mdp.foot_height_per_foot_scan,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        "scan_sensor_names": (
          left_foot_raycast_cfg.name,
          right_foot_raycast_cfg.name,
        ),
      },
      history_length=history_len,
      flatten_history_dim=True,
    )

  for group_name in ("actor_history", "critic", "teacher", "privileged"):
    if group_name in cfg.observations:
      terms = cfg.observations[group_name].terms
      terms.pop("phase", None)
      terms.pop("base_height", None)
      terms.pop("joint_acc", None)
      terms.pop("foot_height", None)

  # Not consumed by the actor or critic networks (obs_groups only lists
  # "actor"/"critic") -- rides along in rollout storage purely so
  # rl_ext.TorqueGuidedPPO can regress the actor's mean action toward it.
  # See mdp.pd_action_guidance_target for why this replaces pd_demand_excess
  # as the actual training signal.
  cfg.observations["torque_guidance"] = ObservationGroupCfg(
    terms={
      "target": ObservationTermCfg(
        func=mdp.pd_action_guidance_target,
        # Bounds the regression target to a plausible one-step correction
        # regardless of how chaotic the underlying state is (mid-fall
        # q_err can otherwise demand an arbitrarily large "correction").
        # See mdp.pd_action_guidance_target / rl_ext.TorqueGuidedPPO.
        params={"max_action_delta": 1.0},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.viewer.body_name = "CHEST_P_LINK"

  if "fell_down" in cfg.terminations:
    cfg.terminations["fell_down"].params["minimum_height"] = 0.55
  else:
    cfg.terminations["fell_down"] = TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.55},
    )

  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("CHEST_P_LINK",)

  assert cfg.commands is not None
  cfg.commands.pop("phase", None)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.4
  # vel_ramp_rate stays None: the command teleports at each resample, so the
  # walk-to-stand transition the deployed robot always goes through is never
  # sampled. Untested, not rejected.
  twist_cmd.viz.z_offset = 1.0
  twist_cmd.ranges.lin_vel_x = (-0.1, 0.1)
  twist_cmd.ranges.lin_vel_y = (-0.15, 0.15)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.02)
  if cfg.curriculum is not None:
    # Every curriculum that used to live here was removed after measuring
    # that none of them moved its metric. curriculums.torque_feasibility_progress
    # still exists, unregistered, as an escape hatch.
    pass
  if cfg.curriculum is not None and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.1, 0.1),
        "lin_vel_y": (-0.15, 0.15),
        "ang_vel_z": (-0.3, 0.3),
      },
      {
        "step": 500 * 48,
        "lin_vel_x": (-0.3, 0.3),
        "lin_vel_y": (-0.25, 0.25),
        "ang_vel_z": (-0.35, 0.35),
      },
      {
        "step": 3000 * 48,
        "lin_vel_x": (-0.3, 0.3),
        "lin_vel_y": (-0.35, 0.35),
        "ang_vel_z": (-0.4, 0.4),
      },
      {
        "step": 7000 * 48,
        "lin_vel_x": (-0.3, 0.3),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.45, 0.45),
      },
    ]

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    if reward_name in cfg.rewards and "asset_cfg" in cfg.rewards[reward_name].params:
      cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["track_linear_velocity"].weight = 3.5
  # Wide kernel: a tight one punishes the COM ripple of a long stride and
  # biases toward short quick steps.
  cfg.rewards["track_linear_velocity"].params["std"] = 0.40
  cfg.rewards["track_angular_velocity"].weight = 3.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.45

  # The only term that says which way to move to fix a bad standing posture;
  # everything else about standing is a verdict, not a direction.
  cfg.rewards["pose"].weight = 1.5
  cfg.rewards["pose"].params["command_name"] = "twist"
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["pose"].params["running_threshold"] = 1.5

  cfg.rewards.pop("soft_landing", None)
  cfg.rewards["impact_vel"] = RewardTermCfg(
    func=mdp.impact_velocity,
    # One of the four safety priorities, so it gets real weight.
    weight=-2.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "limit": 0.15,
      "start_step": 0,
      "pre_contact_limit": 0.45,
      "pre_contact_window_s": 0.1,
      "always_limit": 1.2,
      "command_name": "twist",
      "always_command_threshold": 0.05,
    },
  )
  # One-leg-does-everything gaits are otherwise profitable.
  cfg.rewards["air_time_symmetry"] = RewardTermCfg(
    func=mdp.feet_air_time_symmetry,
    weight=-1.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  # air_time rewards duration aloft, not direction, so a looping backward
  # swing banks the same credit. Constrain the hip, not the knee: real swing
  # is hip-driven and the knee bends passively.
  cfg.rewards["swing_hip_direction"] = RewardTermCfg(
    func=mdp.swing_hip_direction_penalty,
    weight=-0.3,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=("L_CROTCH_P", "R_CROTCH_P"), preserve_order=True
      ),
      "command_threshold": 0.05,
    },
  )
  # Explicit gait clock (Siekmann et al.), prescriptive rather than reactive.
  # Cadence follows commanded speed, never a curriculum: a curriculum step
  # snaps all 4096 envs at once and collapsed a run on 2026-07-25.
  # swing_duration is a duration, not a ratio, so extra cycle time at low
  # speed goes into double support instead of a one-legged balance.
  cfg.rewards["gait_phase"] = RewardTermCfg(
    func=mdp.gait_phase_tracking,
    weight=2.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      "period_slow": 2.0,
      "period_fast": 1.1,
      "swing_duration": 0.4,
      # Calibrated against the sampled distribution of
      # |v_xy| + |wz| across the command_vel curriculum stages, not against
      # a single axis' maximum: because the angular term alone reaches
      # 0.30-0.45, a command_ref of 0.4 put the *median* env at or past
      # saturation from the second stage onward, so the slow end would
      # essentially never have been visited. 0.7 sits near p90 of the final
      # stage, leaving the interpolation to span the range actually
      # sampled (~p10 0.27 -> ~p90 0.71 late in training).
      "command_ref": 0.7,
      "force_std": 12.0,
      "vel_std": 0.15,
      "command_threshold": 0.1,
    },
  )
  # Ablation 2 (2026-07-31), on top of the min_foot_height fix. Every other
  # term that touches single-support balance is a verdict -- flat_support,
  # standing_single_support and the torque family all say "that was bad"
  # without saying which way to move. This one names the next action: before
  # the clock says foot i swings, get the load off foot i. It rides
  # gait_phase's own clock, so it stays meaningful as the period and duty
  # cycle change with speed.
  #
  # +1.0, not gait_phase's 2.0: the term is bounded by 1 per foot and active
  # ~30% of the time (two feet x a 0.15 window), so ~0.2 realized. It is meant
  # to be directional, not dominant -- abl1 just bought swing_height 0.0069 ->
  # 0.0137 at the cost of ~15% of foot_vel_max, and a term strong enough to
  # reshape the gait could spend that gain back.
  cfg.rewards["preswing_transfer"] = RewardTermCfg(
    func=mdp.preswing_weight_transfer,
    weight=1.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "reward_name": "gait_phase",
      "window": 0.15,
      # 57.64 kg -> 565 N. Only sets where `support` saturates, so it needs to
      # be the real weight and not a round number: too low and flight stops
      # being distinguishable from stance.
      "body_weight": 565.0,
    },
  )
  # stride_frequency_target (reactive, post-hoc period measurement) is no
  # longer registered here: gait_phase already prescribes the same period
  # up front, continuously, and the actor can see it coming via
  # mdp.gait_phase_obs -- a channel stride_frequency_target never had.
  # Keeping both meant two reward terms pulling on the same cycle-time
  # budget from redundant signals. mdp.stride_frequency_target itself is
  # left in place (unused) for reference.
  cfg.rewards["no_double_flight"] = RewardTermCfg(
    func=mdp.no_double_flight_penalty,
    weight=-2.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  # No no_double_flight_weight curriculum (removed 2026-07-26). It ramped
  # -2 -> -5 over three stages on a term that realizes -0.006: with the gait
  # clock's swing_ratio at ~0.23 the schedule already prescribes long double
  # support, so both feet are essentially never airborne together. Hardening a
  # guard that never fires is pure schedule complexity. The guard itself stays,
  # at its constant -2.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.5,
    params={"sensor_name": self_collision_cfg.name},
  )
  # Thresholds = matching QP sDist + 1cm buffer, so the policy stays out of
  # the deployment damper's braking zone.
  for prox_cfg, min_dist in (
    (leg_proximity_cfg, 0.02),
    (knee_proximity_cfg, 0.035),
    (arm_torso_proximity_cfg, 0.04),
    (shoulder_chest_proximity_cfg, 0.01),
    (shoulder_body_proximity_cfg, 0.04),
    (wrist_thigh_proximity_cfg, 0.03),
  ):
    cfg.rewards[prox_cfg.name] = RewardTermCfg(
      func=mdp.leg_proximity_cost,
      weight=-2.0,
      params={"sensor_name": prox_cfg.name, "min_dist": min_dist},
    )
  # Near-zero weight, kept so Metrics/pd_demand_ratio_* keep tracking
  # hardware readiness. The real feasibility signal is an auxiliary loss
  # (rl_ext.TorqueGuidedPPO), which cannot form the timidity loop a reward can.
  cfg.rewards["pd_demand_excess"] = RewardTermCfg(
    func=mdp.pd_demand_excess,
    # Small on purpose: at large weights this made the policy too timid to
    # catch a stumble, and the resulting falls fed the penalty back in.
    weight=-0.02,
    params={
      # Below the projection ratio: a projection supplies no gradient, it just
      # discards the excess, so the pull has to start inside the feasible set.
      # cap 4.0 keeps the observed range on the sloped part (ratio max hits 350).
      "soft_ratio": 0.7,
      "cap": 4.0,
      "ema_dt": 0.04,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["torque_limit_margin"] = RewardTermCfg(
    func=mdp.joint_torque_limit_margin_penalty,
    # -0.032 (was -0.08, 2026-07-27): uniform penalty rescale, see action_jerk.
    weight=-0.032,
    params={
      "soft_ratio": 0.8,
      "power": 2.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=mdp.feet_distance_penalty,
    weight=-0.25,
    params={
      "target_distance": 0.14,
      "max_distance": 0.2,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  cfg.rewards["flat_support"] = RewardTermCfg(
    func=mdp.flat_support_penalty,
    # Four coplanar corners are geometrically reachable; what blocked it was
    # ankle torque penalties costing more than this paid. Those are gone.
    weight=-11.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "required_contacts_per_foot": 4,
      # A corner counts as down within 1 mm of the lowest corner of the same
      # foot, not when the contact sensor fires. Relative z cancels tilt; the
      # geom margin+gap route is unavailable, mujoco_warp rejects a non-zero
      # margin on box-box pairs under MULTICCD.
      "corner_tolerance": 0.001,
    },
  )
  cfg.rewards["standing_single_support"] = RewardTermCfg(
    func=mdp.standing_single_support_penalty,
    weight=-6.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
    func=mdp.joint_torque_rate_l2,
    # -1.6e-5 (was -4e-5, 2026-07-27): uniform penalty rescale, see action_jerk.
    weight=-1.6e-5,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r".*CROTCH_Y.*",
          r".*CROTCH_R.*",
          r".*CROTCH_P.*",
          r".*KNEE.*",
          r".*ANKLE.*",
        ),
      )
    },
  )

  cfg.rewards["pose"].params["std_standing"] = {
    r".*CROTCH_P.*": 0.025,
    r".*CROTCH_R.*": 0.025,
    r".*ANKLE.*": 0.02,
    r".*SHOULDER.*": 0.02,
    r".*ELBOW.*": 0.02,
    r".*WRIST.*": 0.02,
    r".*HEAD.*": 0.02,
    r"^(?!.*CROTCH_P.*)(?!.*CROTCH_R.*)(?!.*ANKLE.*)(?!.*SHOULDER.*)(?!.*ELBOW.*)(?!.*WRIST.*)(?!.*HEAD.*).*$": 0.04,
  }
  # Pose is a gentle pull toward a neutral, safe posture, not a gait-shape
  # constraint -- loosened across the whole body. ANKLE_R stays tighter:
  # lateral ankle stability is the one axis where "safe" and "loose" trade
  # off most directly.
  cfg.rewards["pose"].params["std_walking"] = {
    r".*CROTCH_P.*": 1.0,
    r".*CROTCH_R.*": 0.65,
    r".*CROTCH_Y.*": 0.65,
    r".*KNEE.*": 1.1,
    # Ankles stay close to neutral (unlike the rest of the body): only a
    # slight pitch angulation during walking, roll even tighter.
    r".*ANKLE_P.*": 0.25,
    r".*ANKLE_R.*": 0.15,
    r".*CHEST.*": 0.30,
    r".*SHOULDER_P.*": 0.25,
    r".*SHOULDER_R.*": 0.25,
    r".*SHOULDER_Y.*": 0.15,
    r".*ELBOW.*": 0.20,
    r".*WRIST.*": 0.08,
    r".*HEAD.*": 0.03,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*CROTCH_P.*": 0.85,
    r".*CROTCH_R.*": 0.4,
    r".*CROTCH_Y.*": 0.4,
    r".*KNEE.*": 0.95,
    r".*ANKLE_P.*": 0.6,
    r".*ANKLE_R.*": 0.08,
    r".*CHEST.*": 0.24,
    r".*SHOULDER_P.*": 0.06,
    r".*SHOULDER_R.*": 0.06,
    r".*SHOULDER_Y.*": 0.05,
    r".*ELBOW.*": 0.06,
    r".*WRIST.*": 0.05,
    r".*HEAD.*": 0.05,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("CHEST_P_LINK",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("CHEST_P_LINK",)
  cfg.rewards["upright"].weight = 3.0
  cfg.rewards["upright"].params["std"] = 0.2

  cfg.rewards["body_ang_vel"].weight = -0.5
  # Arm swing / torso counter-rotation is also how the stance leg and upper
  # body balance through a stride, not just unwanted spin.
  cfg.rewards["angular_momentum"].weight = -0.1
  cfg.rewards["angular_momentum"].params["sensor_name"] = "robot/root_angmom"
  cfg.rewards["dof_pos_limits"].weight = -1.0
  # One consolidated torque term instead of three overlapping ones. Ankles
  # carry the emphasis: weakest leg actuators, and low ankle effort goes with
  # a flat foot rather than against it.
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_weighted_l2,
    weight=-1.2e-5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      # Keep ANKLE_R low: holding the sole flat against the ankle-roll droop
      # needs sustained roll torque, and taxing it is what stalled flat_support.
      "coeffs": {r"ANKLE_R": 1.5, r"ANKLE_P": 2.0},
    },
  )
  # air_time retired: the gait clock owns swing duration, and the actor can
  # see the clock coming through gait_phase_obs. Watch peak_height_mean
  # against air_time_mean -- long air time with a flat trajectory means the
  # foot is skimming, which gait_phase scores more leniently than a contact
  # sensor would.
  cfg.rewards.pop("air_time", None)
  # Height shaping lives only in the min_foot_height floor below; per-step
  # clearance/swing-height taxes made short fast shuffling optimal.
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_swing_height", None)
  # Clock-gated, not landing-triggered. Charging per landing means never
  # leaving the ground costs nothing, so the term meant to make the robot
  # lift its foot was the largest single penalty on lifting it.
  cfg.rewards["min_foot_height"] = RewardTermCfg(
    func=mdp.clock_swing_height_deficit,
    weight=-5.0,
    params={
      # True sole clearance: the foot sites sit in the sole plane.
      "min_height": 0.015,
      "reward_name": "gait_phase",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  cfg.rewards["foot_slip"].func = mdp.split_feet_slip
  cfg.rewards["foot_slip"].weight = -28.0
  cfg.rewards.pop("action_acc_l2", None)
  
  # Smoothness is two terms, not five: action_jerk (second difference, so an
  # ample stride is cheap and a tremor is not) and standing_joint_vel,
  # command-gated so it never taxes the gait. Penalising the first difference
  # can only ask for less motion, never for soft motion.
  cfg.rewards.pop("action_rate_l2", None)
  cfg.rewards.pop("stance_action_acc_l2", None)
  cfg.rewards.pop("upper_body_action_acc_l2", None)
  cfg.rewards.pop("leg_joint_acc_l2", None)
  cfg.rewards["action_jerk"] = RewardTermCfg(
    func=mdp.action_jerk_l2,
    # Second difference, so it targets tremor and is blind to slow sway;
    # that one is standing_base_motion's job, not a bigger weight here.
    weight=-45.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "coeffs": {r"CHEST|HEAD|SHOULDER|ELBOW|WRIST": 25.0},
    },
  )
  # Legs only: hip yaw and roll set where the feet point, ankle roll whether
  # they land flat. Hip and knee pitch stay out, so this prescribes a stance
  # without prescribing a squat depth.
  cfg.rewards["standing_pose"] = RewardTermCfg(
    func=mdp.standing_pose_penalty,
    weight=-40.0,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r".*CROTCH_Y.*", r".*CROTCH_R.*", r".*ANKLE_R.*"),
      ),
    },
  )
  cfg.rewards["standing_joint_vel"] = RewardTermCfg(
    func=mdp.standing_joint_vel_l2,
    # An L2 sum, so it only bites on fast residual motion; slow sway is
    # standing_base_motion.
    weight=-0.7,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # L1 by design: the terms that could have caught slow drift at zero command
  # all go soft in exactly that regime.
  cfg.rewards["standing_base_motion"] = RewardTermCfg(
    func=mdp.standing_base_motion,
    weight=-1.5,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "ang_weight": 0.5,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # No air_time_target curriculum any more (removed 2026-07-25, alongside
  # making gait_phase's cadence speed-dependent). It used to ramp
  # threshold_max 0.5 -> 0.85s, i.e. keep demanding ever-longer swings; the
  # gait clock now prescribes swing duration outright (0.4s at every speed,
  # see gait_phase above), so a ceiling climbing toward 0.85s is no longer
  # "more ambition", it is a second, contradictory timing target pulling
  # against the clock. Same reasoning that retired stride_frequency_target:
  # once an explicit clock owns the timing, the post-hoc measurements of
  # that timing should stop trying to steer it.
  #
  # air_time itself stays, at a fixed threshold_max just above the
  # prescribed swing, in a narrower role -- gait_phase's swing term scores
  # exp(-(F/force_std)^2), which a foot merely *unloaded* to ~10N still
  # collects ~90% of without ever breaking contact. air_time reads the
  # binary contact sensor, so it remains the only term requiring a genuine
  # lift-off rather than a light-footed drag.
  cfg.curriculum.pop("air_time_target", None)
  cfg.rewards["foot_slip"].params["sensor_name"] = feet_ground_split_cfg.name
  cfg.rewards["foot_slip"].params["command_name"] = "twist"
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.1
  cfg.rewards["foot_slip"].params["standing_scale"] = 4.0

  cfg.rewards["termination_penalty"] = RewardTermCfg(func=mdp.is_terminated, weight=-2000.0)

  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
      "left_foot1_collision",
      "left_foot2_collision",
      "left_foot3_collision",
      "left_foot4_collision",
      "right_foot1_collision",
      "right_foot2_collision",
      "right_foot3_collision",
      "right_foot4_collision",
    )
    # Range is set further down, to (0.4, 1.0). Nothing here.
  # push_robot comes from the base velocity config, not from this file.
  cfg.events.pop("push_robot", None)

  # --- Raw-torque peak: one observable, gradient-carrying term replacing the
  # magnitude family (Leo, 2026-08-04) ------------------------------------------
  #
  # The three terms popped below all charge torque *magnitude* by different
  # proxies, none of which the actor can see. Realized at iteration 3500 of
  # 2026-08-04_09-27-15: pd_demand_excess -0.38, torque_limit_margin -0.51,
  # joint_torques_l2 -0.92, i.e. -1.81 together. That is the budget the new term
  # has to take over, and the number to calibrate its weight against.
  #
  # NOT popped, on purpose: joint_torque_rate_l2 (-0.99 realized) charges
  # d(tau)/dt, not |tau|. It is the only thing standing between the joints and
  # chatter, and action_jerk does not cover it -- action_jerk acts on the position
  # target, and at a frozen target a moving robot still swings its torque through
  # the kd term. Removing it was proposed and rejected.
  #
  # The definitions above are left in place rather than deleted: they carry the
  # history of two collapses and several regrades, and popping keeps this trivially
  # reversible.
  for _dead in ("pd_demand_excess", "torque_limit_margin", "joint_torques_l2"):
    cfg.rewards.pop(_dead, None)

  cfg.rewards["raw_torque_peak"] = RewardTermCfg(
    func=mdp.raw_torque_peak_penalty,
    # Start almost silent and let the curriculum below raise it. Sizing this a
    # priori is not possible: the cost is sum_j clamp(peak_j - 0.7, 0, 4) and
    # nothing yet measures that sum. Read Episode_Reward/raw_torque_peak at the
    # first milestones and re-grade against the -1.81 the popped terms realized.
    # Re-graded for the log1p shape at threshold 1.0 (2026-08-05): the previous
    # schedule was calibrated against clamp(excess, 0, 4) at threshold 0.7, a
    # different cost scale entirely, so its numbers do not carry over. Read
    # Episode_Reward/raw_torque_peak at the first milestones and re-grade again.
    weight=-0.05,
    params={
      # 1.0, the effort limit itself: charge exactly what the motor cannot deliver.
      # No cap -- log1p bounds the growth instead, see the term's docstring.
      "soft_ratio": 1.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  assert cfg.curriculum is not None

  # Progressive, as asked: the term is nearly silent while the gait forms, then
  # takes over the magnitude budget. Ramped over 6000 iterations because a torque
  # penalty applied early is the documented way to make this policy timid -- it
  # suppresses the large corrective actions that catch a stumble, whose own
  # chaotic demand then feeds the penalty. Steps are in env steps: iteration x 48.
  cfg.curriculum["raw_torque_peak_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "raw_torque_peak",
      # Pushed further than the previous schedule stopped. Measured on
      # 2026-08-04_22-xx: nothing moved at all until -0.40, at which point the
      # demand fell hard (peak_mean 3.35 -> 1.93, over_limit 0.62 -> 0.49) with
      # zero falls and upright at its best -- so -0.60 was a ceiling set by caution,
      # not by evidence. The brief spike in falls at -0.20 (0.11-0.13 around
      # iteration 3500-4000) was transitional and cleared by itself.
      "weight_stages": [
        {"step": 0, "weight": -0.05},
        {"step": 1000 * 48, "weight": -0.15},
        {"step": 2500 * 48, "weight": -0.35},
        {"step": 4000 * 48, "weight": -0.65},
        {"step": 6000 * 48, "weight": -1.00},
        {"step": 8000 * 48, "weight": -1.50},
      ],
    },
  )

  # standing_envs is a constant 0.4, set directly on the command term above --
  # it stopped being a curriculum when the decay was removed (both prior runs
  # progressively forgot how to stand as standing exposure dropped). A
  # single-stage curriculum is just an indirection.

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations[actor_group_name].enable_corruption = False
    # push_base is dropped in rhps1_flat_env_cfg, not here: it is added later
    # in this function, so a pop at this point would see nothing.
    # Disable debug visualizers to recover viewer FPS.
    twist_cmd.debug_vis = False
    for sensor in cfg.scene.sensors:
      if isinstance(sensor, RayCastSensorCfg):
        sensor.debug_vis = False

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  # 20000/400 is an emulated position servo, not a measurement -- the robot
  # hides its own P/PI loop in the drive. Per joint, so a mismatch between
  # joints is covered too.
  cfg.events["actuator_gains"] = EventTermCfg(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={"stiffness_range": (0.85, 1.15), "damping_range": (0.85, 1.15)},
  )

  # Per env and per episode. shared_random=False is the real change: with True
  # all 4096 envs saw one ground, which randomises nothing. Range widened down
  # to 0.4, the slippery case that makes it fall and was never sampled.
  cfg.events["foot_friction"].mode = "reset"
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.0)
  cfg.events["foot_friction"].params["shared_random"] = False

  # Mass and inertia together: body_mass alone leaves the inertia tensor, which
  # models a point mass rather than a density change.
  # alpha is a LOGARITHM -- mass scales by exp(2*alpha), default (0.0, 0.0).
  # Written as (0.9, 1.1) once, it made every body 6 to 9x its weight and
  # killed a whole run. +/-10% is ln(1.1)/2 and ln(0.9)/2.
  cfg.events["link_inertia"] = EventTermCfg(
    func=mdp.dr.pseudo_inertia,
    mode="startup",
    params={"alpha_range": (-0.0527, 0.0477), "asset_cfg": SceneEntityCfg("robot")},
  )

  # All bodies, not just the torso. Tighter range because the segments are
  # much smaller.
  cfg.events["link_com"] = EventTermCfg(
    func=mdp.dr.body_com_offset,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "add",
      "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
    },
  )

  # The only term that tests recovery rather than tracking. Zero to two hits
  # per 20 s episode.
  cfg.events["push_base"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(8.0, 12.0),
    params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
  )

  # Episodes used to start exactly at q0, so the policy never had to recover an
  # imperfect initial posture -- which is the case when it is armed on the robot.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

  # The QP solves at 200 Hz on the robot against 1 kHz in validation sim, and is
  # not guaranteed stable under CPU load. Damping follows as 2*sqrt(K) in the
  # actuator, like mc_rtc: one plant randomised, not two independent gains.
  cfg.events["posture_filter"] = EventTermCfg(
    func=mdp.randomize_posture_task_stiffness,
    mode="reset",
    params={"stiffness_range": (0.75, 1.25)},
  )

  # A per-episode constant bias, not per-step noise: a five-frame history
  # averages zero-mean noise away but stays credulous to an offset, which is
  # the shape of the symptom on the robot. base_lin_vel is the serious channel
  # (+0.0167 m/s measured while walking, zero at rest); projected_gravity is
  # small because the estimated attitude agrees with the encoders to 0.12 deg.
  # No base_ang_vel: a gyro bias drifts yaw, it does not tilt.
  cfg.events["sensor_bias"] = EventTermCfg(
    func=mdp.randomize_sensor_bias,
    mode="reset",
    params={
      "bias_ranges": {
        "base_lin_vel": (-0.05, 0.05),
        "projected_gravity": (-0.01, 0.01),
      }
    },
  )

  # Sole height straight off the sites. It replaces peak_height_mean, which
  # reset on the first corner touching while landing fires on all four, so it
  # read ~70x low. Depends on no reward term, so nothing can silence it.
  # Does the robot advance? Nothing else logged here answers that.
  cfg.metrics["command_progress"] = MetricsTermCfg(
    func=mdp.log_command_progress,
    params={"command_name": "twist", "command_threshold": 0.05},
  )

  cfg.metrics["sole_height"] = MetricsTermCfg(
    func=mdp.log_sole_height,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
  )

  _apply_policy0_baseline(cfg)
  return cfg


def _apply_policy0_baseline(cfg: ManagerBasedRlEnvCfg) -> None:
  """Ramene la configuration a la policy 0, plus les ameliorations retenues.

  Decide avec Leo le 2026-08-07. La policy 0 (run 2026-07-10_20-59-17) est la
  seule qui ait marche sur le robot reel ; tout ce qui a ete ajoute depuis a ete
  valide en simulation, jamais sur du materiel. Plutot que de defaire trente
  reglages disperses dans ce fichier, la bascule est ici, en un seul bloc
  lisible d'un coup.

  Ce qui est REPRIS d'abl15 :
    - armatures reelles (rhps1_constants, hors joints a verins)
    - standing_pose -40, le seul terme nouveau valide isolement
    - flat_support corrige et impact_vel renforce : les deux defauts que Leo
      avait releves en jouant les checkpoints (deroule talon-pointe, vitesse
      d'impact)
    - les six termes de proximite : demande explicite, c'est de la securite
    - action_jerk, qui remplace action_rate_l2 / stance_action_acc_l2 /
      upper_body_action_acc_l2
    - mirror loss

  Ce qui est REPOUSSE :
    - l'echelle d'action x4.67 (voir _POLICY0_LEG_SCALE) et donc raw_torque_peak
      avec elle : cette penalite n'a existe que parce que l'echelle avait
      grossi. Petite echelle, petit probleme de couple.
    - gait_phase, air_time_symmetry, preswing_transfer, swing_hip_direction :
      jamais isoles, et gait_phase agrandit l'observation.
    - l'observation V5 (566) et V4 (266).

  L'effet le plus concret est la derniere ligne : l'observation redevient
  EXACTEMENT la V3 a 126 dimensions, celle que `utils.cpp` case 0 sait deja
  construire. Aucun travail C++ pour deployer ce run.
  """
  # Both names live in rhps1_rough_env_cfg. Asserted below rather than assumed:
  # a wrong sensor name would give a silent reward.
  _SPLIT_SENSOR = "feet_ground_contact_split"
  site_names = ("left_foot", "right_foot")
  assert any(getattr(s, "name", None) == _SPLIT_SENSOR for s in cfg.scene.sensors), (
    f"capteur {_SPLIT_SENSOR!r} absent de la scene"
  )

  # Terms that go.
  for name in (
    "raw_torque_peak",
    "gait_phase",
    "air_time_symmetry",
    "preswing_transfer",
    "swing_hip_direction",
  ):
    cfg.rewards.pop(name, None)
  cfg.curriculum.pop("raw_torque_peak_weight", None)

  # Policy 0 terms with no successor.
  cfg.rewards["air_time"] = RewardTermCfg(
    func=mdp.split_feet_air_time,
    weight=2.0,
    params={
      "sensor_name": _SPLIT_SENSOR,
      "command_name": "twist",
      "threshold_min": 0.01,
      "threshold_max": 0.2,
      "command_threshold": 0.1,
      "overflow_threshold": 2.0,
      "power": 2.0,
      # Zero: the landing cost exactly cancelled the airborne payment, making a
      # short step net negative -- a lift reward that punished small lifts.
      "touchdown_cost": 0.0,
    },
  )
  # The only torque penalty left once raw_torque_peak goes.
  cfg.rewards["torque_limit_margin"] = RewardTermCfg(
    func=mdp.joint_torque_limit_margin_penalty,
    weight=-0.16,
    params={"soft_ratio": 0.8, "power": 2.0},
  )
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2, weight=-1e-5
  )

  # min_foot_height reads the gait clock, so it goes with gait_phase. The two
  # terms below are what policy 0 carried instead, and they worked on the robot.
  cfg.rewards.pop("min_foot_height", None)
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=mdp.split_feet_swing_height,
    weight=-5.0,
    params={
      "sensor_name": _SPLIT_SENSOR,
      "target_height": 0.15,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  # Cost is |z - target| * horizontal speed. With the target far above what the
  # foot reaches, the delta is constant and the term degenerates into a tax on
  # foot speed. The weight here is extrapolated, not measured: check
  # Episode_Reward/foot_clearance at the first milestone.
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.feet_clearance_velocity_weighted,
    weight=-35.0,
    params={
      "target_height": 0.04,
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # --- observation : retour a la V3, 126 dimensions ----------------------
  actor = cfg.observations["actor"].terms
  for name in ("gait_phase", "raw_torque"):
    actor.pop(name, None)
  # Actions with history 5: the velocity-target EMA (alpha 0.8) and the
  # feasibility projection are both hidden state. Observation goes 126 -> 246,
  # which needs its own case in utils.cpp.
  if "actions" in actor:
    actor["actions"].history_length = 5

  # Finite difference, like the robot: mc_rtc gets no joint velocities on RHPS1
  # and derives the encoders.
  if "joint_vel" in actor:
    actor["joint_vel"] = ObservationTermCfg(
      func=mdp.joint_vel_encoder_finite_difference,
      # Per-sample encoder noise. RHPS1 encoders quantise at 6e-7 rad on the
      # joint side, so differentiating amplifies almost nothing; 0.001 put 21x
      # too much noise on the channel, measured against the real-robot log.
      params={"encoder_noise": 0.00005},
      noise=None,
    )
  # The critic keeps its privileged channels, they cost nothing at deployment.
  # gait_phase must go, it references a reward that no longer exists.
  critic = cfg.observations["critic"].terms
  critic.pop("gait_phase", None)
  # torque_guidance existed only for torque_guidance_coef, now zero.
  cfg.observations.pop("torque_guidance", None)


def rhps1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create RHPS1 flat terrain velocity configuration."""
  cfg = rhps1_rough_env_cfg(play=play)

  # Flat terrain can use a lighter contact configuration to reduce memory.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 96

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  assert cfg.curriculum is not None
  if "terrain_levels" in cfg.curriculum:
    del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.3, 0.3)
    twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
    twist_cmd.ranges.ang_vel_z = (-0.45, 0.45)

    # Removed here, where the whole config exists: the play block in the rough
    # config runs before push_base is added, and named push_robot anyway.
    cfg.events.pop("push_base", None)

  return cfg


def rhps1_stepping_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Simplified RHPS1 config focused on learning to lift feet and step."""
  cfg = rhps1_flat_env_cfg(play=play)

  if play:
    return cfg

  cfg.episode_length_s = 10.0

  if cfg.curriculum is not None and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.1, 0.1),
        "lin_vel_y": (-0.1, 0.1),
        "ang_vel_z": (-0.2, 0.2),
      },
    ]

  # This config wants an unpushed robot; the old pop named push_robot.
  cfg.events.pop("push_base", None)

  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0

  return cfg
