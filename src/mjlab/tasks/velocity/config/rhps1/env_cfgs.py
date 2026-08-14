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
    # Ce bloc ne desactive plus que la corruption, la visualisation et la duree
    # d'episode. Ce qu'il contenait avant et qui ne faisait rien :
    #   pop("teacher"), pop("air_time_weight"), pop("standing_envs"),
    #   pop("push_robot") et la branche "actor_history"
    # Aucun de ces cinq noms n'est cree nulle part dans ce fichier, et le
    # ", None" des pop rendait l'echec muet. push_base, lui, existe -- il est
    # retire dans rhps1_flat_env_cfg, seul endroit ou toute la config est deja
    # construite : ajoute plus bas dans cette fonction, il est invisible ici.
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

  # Randomisation des gains PD. Les 20000/400 ne sont pas une mesure : ce sont
  # les valeurs d'un servo de position emule, le vrai robot cachant sa boucle
  # P/PI dans le drive. Une politique entrainee sur une seule valeur apprend la
  # reponse exacte de ce PD-la ; la randomiser l'oblige a rester correcte sur
  # une famille de plants.
  #
  # +/-15 % et par articulation independamment : chaque drive a son propre
  # reglage, donc un desaccord entre articulations est le cas realiste, et c'est
  # aussi le plus exigeant.
  cfg.events["actuator_gains"] = EventTermCfg(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={"stiffness_range": (0.85, 1.15), "damping_range": (0.85, 1.15)},
  )

  # --- Elargissement de la randomisation, 2026-08-12 -----------------------
  #
  # L'audit du 2026-08-12 a montre que la randomisation reelle etait plus
  # etroite que ce qu'on croyait : pas de masse, pas d'inertie, un decalage de
  # centre de masse limite au torse, aucune perturbation externe, et une
  # friction tiree une seule fois au demarrage et PARTAGEE par les 4096
  # environnements -- donc une seule valeur de friction pour tout un run.
  #
  # Principe applique ici : randomiser ce qu'on n'a pas identifie, et avec des
  # plages qui restent physiquement plausibles. Ce n'est pas une couverture
  # maximale, c'est une couverture honnete.

  # Friction par environnement et par episode, au lieu d'une valeur unique
  # tiree au demarrage. shared_random=False est le vrai changement : avec True,
  # les 4096 environnements voyaient le meme sol, ce qui ne randomise rien du
  # point de vue de la politique. Plage elargie vers le bas (0.4) : un sol plus
  # glissant que 0.5 est le cas qui fait tomber, et il n'etait jamais echantillonne.
  cfg.events["foot_friction"].mode = "reset"
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.0)
  cfg.events["foot_friction"].params["shared_random"] = False

  # Masse ET inertie ensemble, via pseudo_inertia : dr.body_mass seul laisse le
  # tenseur d'inertie inchange, ce qui modelise une masse ponctuelle au centre
  # de masse plutot qu'un changement de densite -- le code de mjlab emet
  # d'ailleurs un avertissement explicite a ce sujet. alpha_range scale les deux
  # de facon coherente. +/-10 % : l'incertitude plausible sur une masse de
  # segment mesuree au CAO, pas une variation reelle entre exemplaires.
  #
  # ATTENTION : alpha est un LOGARITHME, la masse et l'inertie sont multipliees
  # par exp(2*alpha), et le defaut de la fonction est (0.0, 0.0). Ecrit (0.9,
  # 1.1) le 2026-08-12 en le prenant pour un multiplicateur, ce qui donnait
  # exp(1.8)=6.05 a exp(2.2)=9.03 : chaque corps pesait six a neuf fois son
  # poids, le robot faisait 350 a 520 kg au lieu de 57.6, et il s'effondrait en
  # 1.4 s quoi que fasse la politique. Le run 2026-08-12_16-04-55 est mort de
  # ca -- longueur d'episode figee a 70 pas de l'iteration 50 a 492, zero
  # time_out, alors que la reference tenait 3700 pas.
  #
  # +/-10 % s'ecrit donc alpha = ln(1.1)/2 = 0.0477 et ln(0.9)/2 = -0.0527.
  cfg.events["link_inertia"] = EventTermCfg(
    func=mdp.dr.pseudo_inertia,
    mode="startup",
    params={"alpha_range": (-0.0527, 0.0477), "asset_cfg": SceneEntityCfg("robot")},
  )

  # Le decalage de centre de masse existait deja mais ne portait que sur le
  # torse. L'etendre a tous les corps couvre l'erreur d'assemblage et de
  # modelisation ailleurs que dans le tronc, avec une plage plus serree (+/-1 cm)
  # puisqu'elle s'applique maintenant a des segments bien plus petits.
  cfg.events["link_com"] = EventTermCfg(
    func=mdp.dr.body_com_offset,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "add",
      "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
    },
  )

  # Poussee externe : absente jusqu'ici alors que c'est le terme standard de la
  # litterature locomotion, et le seul qui teste la recuperation plutot que le
  # suivi. Impulsion instantanee sur la base, toutes les 8 a 12 s -- donc zero
  # a deux fois par episode de 20 s. 0.4 m/s lateral est un desequilibre net
  # sans etre une chute programmee.
  cfg.events["push_base"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(8.0, 12.0),
    params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
  )

  # Posture de depart : les offsets articulaires au reset etaient a (0, 0),
  # donc chaque episode demarrait exactement a q0. Une politique entrainee
  # ainsi n'a jamais a rattraper une posture initiale imparfaite, ce qui est
  # pourtant le cas au moment de l'armement sur le robot. +/-0.05 rad est petit
  # devant l'amplitude d'un pas et suffit a supprimer cette hypothese.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

  # Raideur du filtre de PostureTask, donc du retard vu par la politique. Le
  # taux de resolution du QP vaut 200 Hz sur le robot contre 1 kHz en
  # simulation de validation et n'est pas garanti stable sous charge CPU ;
  # +/-25 % couvre cette incertitude. L'amortissement suit en 2*sqrt(K) dans
  # l'actionneur, comme mc_rtc le fait : c'est un plant qu'on randomise, pas
  # deux gains independants.
  cfg.events["posture_filter"] = EventTermCfg(
    func=mdp.randomize_posture_task_stiffness,
    mode="reset",
    params={"stiffness_range": (0.75, 1.25)},
  )

  # Biais capteur constants sur l'episode (2026-08-12). Motive par une
  # observation sur le robot : il se tient systematiquement sur l'arriere et
  # tombe en arriere des qu'on lui commande une vitesse negative, alors que la
  # simulation est saine.
  #
  # Ce qui a ete elimine par la mesure avant d'en arriver la : la posture
  # nominale est centree dans le polygone de sustentation (CoM x=-16.4 mm,
  # talon -129.6, pointe +105.4, soit 48 % depuis le talon) ; les plages de
  # commande sont symetriques ((-0.3, 0.3) en x au niveau max) ; les butees
  # articulaires laissent 1.11 rad d'un cote et 0.94 de l'autre a la cheville.
  # Aucune de ces trois pistes ne produit d'asymetrie avant/arriere.
  #
  # Reste une asymetrie qui ne vit pas dans le modele mais dans la mesure. Le
  # bruit d'observation existant est recentre a chaque pas ; un estimateur
  # reel, lui, se trompe dans la meme direction pendant tout l'essai. Avec cinq
  # pas d'historique la politique moyenne le premier et reste entierement
  # credule face au second, ce qui est exactement la forme du symptome :
  # systematique et toujours du meme cote.
  #
  # Chiffres revus le 2026-08-12 contre le log ROBOT REEL 2026-08-10-17-08
  # (200 Hz ; mc_mujoco tourne a 1 kHz), qui a confirme l'un des deux canaux et
  # infirme l'autre.
  #
  # base_lin_vel : confirme, et c'est le canal serieux. Sur 30 s de marche la
  # pose estimee par MCWaiko bouge de -0.017 m pendant que sa propre vitesse
  # integree donne +0.502 m -- deux sorties du meme observateur qui divergent
  # d'un facteur 30. Le biais vaut +0.0167 m/s en marche (0.0167 * 30 s = 0.502
  # m, la divergence se referme exactement) contre +0.0002 m/s a l'arret : le
  # biais nait du mouvement, il n'existe pas au repos. On garde 0.05, soit
  # trois fois le mesure : la mesure porte sur un seul essai, a une seule
  # allure, sur un sol propre, et le biais depend justement du mouvement.
  #
  # projected_gravity : infirme, et ramene de 0.05 a 0.01. Le -0.0138 lu a
  # l'arret n'est pas une erreur d'estimation, c'est un vrai tangage du bassin
  # : les encodeurs donnent -0.909 deg (soit -0.0159 en gravite projetee) pour
  # la meme posture, l'estimateur annonce -0.0138, ils s'accordent a 0.12 deg.
  # L'assiette est donc juste et 0.05 valait vingt-cinq fois l'erreur reelle.
  # On garde 0.01 (~0.6 deg) parce que cet accord ne prouve qu'une coherence
  # interne : il suppose les semelles a plat sur un sol de niveau, et un sol
  # incline decalerait les deux estimations ensemble sans qu'on le voie.
  #
  # base_ang_vel volontairement absent bien que la fonction l'accepte : un
  # biais gyrometrique produit une derive en lacet, pas une inclinaison, et il
  # serait de toute facon noye sous le bruit de +/-0.3 rad/s deja present.
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

  # Hauteur de semelle, mesuree directement sur les sites. Elle remplace
  # Metrics/peak_height_mean, qui sous-estimait d'un facteur ~70 : ce pic-la
  # etait remis a zero des le premier coin qui touche alors que l'atterrissage se
  # declenche sur les quatre, donc chaque atterrissage reel produisait une bonne
  # valeur suivie de plusieurs zeros. Le 2026-08-08 elle annoncait 0.0005 m la ou
  # la mesure directe sur le meme checkpoint donnait 0.037.
  #
  # Ne depend d'aucun terme de recompense : la retirer ou la changer ne peut plus
  # eteindre l'instrument, ce qui est exactement ce qui est arrive ce jour-la.
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
  # Le capteur par pied est cree dans rhps1_rough_env_cfg, hors de portee ici.
  # Verifie plutot que devine : un nom faux donnerait une recompense muette.
  _SPLIT_SENSOR = "feet_ground_contact_split"
  # Idem : local a rhps1_rough_env_cfg. Ce sont les sites plantes dans le plan de
  # la semelle depuis le 2026-07-26, pas les anciens 20 mm plus haut.
  site_names = ("left_foot", "right_foot")
  assert any(getattr(s, "name", None) == _SPLIT_SENSOR for s in cfg.scene.sensors), (
    f"capteur {_SPLIT_SENSOR!r} absent de la scene"
  )

  # --- termes qui partent ------------------------------------------------
  for name in (
    "raw_torque_peak",
    "gait_phase",
    "air_time_symmetry",
    "preswing_transfer",
    "swing_hip_direction",
  ):
    cfg.rewards.pop(name, None)
  cfg.curriculum.pop("raw_torque_peak_weight", None)

  # --- termes de la policy 0 qui n'ont pas de successeur -----------------
  # air_time: en retirant gait_phase on retire la seule chose qui prescrivait
  # une duree de vol. Sans lui, plus rien ne demande de lever le pied et le
  # robot converge vers le trainage. La policy 0 le portait a +2.
  # Parametres repris tels quels du run 2026-07-10_20-59-17, pas re-inventes.
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
      # 0.0 (etait 0.15, 2026-08-08). Mesure sur 2026-08-07_15-40-43, iterations
      # 6000-7200 : ce terme realisait +0.0008 pour un poids de +2, c'est-a-dire
      # rien. Le cout d'atterrissage annulait exactement le paiement du vol, et
      # rendait meme un pas court net negatif -- une recompense de levee qui
      # punissait les petites levees.
      "touchdown_cost": 0.0,
    },
  )
  # torque_limit_margin: raw_torque_peak partant, il ne resterait aucune
  # penalite de couple. C'est celle que la policy 0 portait, a -0.16.
  cfg.rewards["torque_limit_margin"] = RewardTermCfg(
    func=mdp.joint_torque_limit_margin_penalty,
    weight=-0.16,
    params={"soft_ratio": 0.8, "power": 2.0},
  )
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2, weight=-1e-5
  )

  # min_foot_height doit partir avec gait_phase : clock_swing_height_deficit lit
  # la phase de l'horloge, et sans elle il leve au premier pas. Il n'y aurait
  # alors plus rien du tout sur la hauteur de pied. On remet les deux termes que
  # la policy 0 portait a sa place -- ce sont eux qui ont marche sur le robot.
  #
  # A garder en tete pour la suite : le piege documente de min_foot_height (il
  # punissait la tentative de pas et payait zero pour rester debout) portait sur
  # CETTE forme-la, pas sur celles-ci.
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
  # Cible 0.15 -> 0.04 et poids -4 -> -35 (2026-08-08).
  #
  # Le cout est |z - cible| * vitesse_horizontale. A 0.15 contre 0.037 reellement
  # atteint (mesure directe sur le checkpoint 7050), delta valait ~0.13 quoi que
  # fasse le robot : le terme degenerait en "penalise la vitesse horizontale du
  # pied", donc il taxait la marche au lieu de faconner la hauteur. A 0.04 il
  # varie entre 0.003 et 0.039 et redevient un gradient en hauteur.
  #
  # Le poids est EXTRAPOLE, pas mesure : delta chute d'un facteur ~6.5, donc a
  # poids constant le realise passerait de -0.11 a ~-0.017, c'est-a-dire inerte.
  # -35 viserait ~-0.15, du niveau de standing_base_motion, present sans dominer.
  # C'est le seul chiffre de ce run que je n'ai pas su deriver d'une mesure : a
  # verifier sur Episode_Reward/foot_clearance au premier jalon et a corriger tot.
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
  # actions AVEC historique 5, comme les dernieres politiques. Deux raisons, pas
  # une : l'actionneur porte une EMA sur la cible de vitesse (alpha 0.8) dont
  # rien d'autre dans l'observation ne revele l'etat, et la projection de
  # faisabilite fait que plusieurs actions brutes donnent la meme execution --
  # une politique qui ne voit que ce qu'elle a demande ne peut pas les
  # distinguer. Les deux mecanismes sont actifs dans ce run.
  #
  # Cout au deploiement : l'observation passe de 126 a 246 dimensions. Ce n'est
  # ni la V3 ni la V4 (266, qui ajoute gait_phase), donc il faut un cas de plus
  # dans utils.cpp -- mais c'est le corps de case 2 sans sa derniere ligne.
  if "actions" in actor:
    actor["actions"].history_length = 5

  # joint_vel par difference finie, comme le robot reel. Voir la docstring de
  # joint_vel_encoder_finite_difference : mc_rtc ne recoit aucune vitesse
  # articulaire sur RHPS1 et derive les encodeurs. Le bruit blanc de +/-1.5 est
  # retire, il est desormais produit par la derivation elle-meme.
  if "joint_vel" in actor:
    actor["joint_vel"] = ObservationTermCfg(
      func=mdp.joint_vel_encoder_finite_difference,
      # 0.001 rad, pas les 0.01 du terme joint_pos : ce parametre modelise le
      # bruit d'encodeur PAR ECHANTILLON, et sur une articulation reduite 210:1
      # il est petit. La part lente (calibration, flexion) est deja portee par
      # l'evenement encoder_bias, et un biais constant disparait de toute facon
      # dans une difference. Mesure sur 64 envs : la corruption resultante de la
      # vitesse vaut 1.03 rad/s d'ecart-type, dont 1.02 vient de la difference
      # finie elle-meme et non du bruit -- c'est bien la derivation qui domine,
      # comme sur le robot. A 0.01 on monterait a 1.93, soit deux fois le signal.
      # 5e-5, pas 0.001 (2026-08-12). Mesure contre le log robot reel
      # 2026-08-10-17-08 : a l'arret, la vitesse articulaire vue par le
      # controleur a un ecart-type de 0.0079 rad/s. La meme mesure en
      # simulation, apres stabilisation, donnait 0.1628 avec 0.001 -- soit 21
      # fois trop de bruit sur un canal entier d'observation, que la politique
      # aurait appris a ignorer purement et simplement.
      #
      # La raison est physique : les encodeurs de RHPS1 ont un pas de
      # quantification de 6e-7 rad cote articulation, donc les deriver
      # n'amplifie presque rien. 0.001 rad valait 1600 fois le pas reel. Le
      # raisonnement de la docstring sur l'anti-correlation reste juste, mais
      # l'effet est invisible ici : l'autocorrelation a lag 1 du signal reel
      # vaut +0.978, c'est du mouvement, pas du bruit derive.
      #
      # Etalonnage mesure : 0.001 -> 0.163, 2e-4 -> 0.034, 1e-4 -> 0.017,
      # 5e-5 -> 0.0097, 0 -> 0.0041. On garde 5e-5 plutot que 4e-5 pile : la
      # cible vient d'un seul essai sur un seul robot, une marge de 1.2x est
      # raisonnable, et le plancher a bruit nul (0.0041) montre que la moitie
      # du bruit reel n'est de toute facon pas d'origine encodeur.
      params={"encoder_noise": 0.00005},
      noise=None,
    )
  # Le critic peut garder ses canaux privilegies (foot_height_scan,
  # joint_torques) : ils n'existent qu'a l'entrainement et ne coutent rien au
  # deploiement. Seul gait_phase doit partir, il reference une recompense qui
  # n'existe plus.
  critic = cfg.observations["critic"].terms
  critic.pop("gait_phase", None)
  # Le groupe torque_guidance n'existait que pour torque_guidance_coef, mis a
  # zero apres cinq echecs et retire de rl_cfg le meme jour. Le laisser ferait
  # calculer une observation entiere que plus personne ne lit.
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

    # Les poussees doivent partir en play, et jusqu'ici elles restaient. Deux
    # raisons cumulees, toutes deux silencieuses :
    #   - le bloc play du rough cfg fait pop("push_robot"), or l'evenement
    #     s'appelle push_base ; le ", None" avale l'absence sans rien dire ;
    #   - meme avec le bon nom ce serait inutile, ce bloc-la s'execute avant la
    #     ligne qui ajoute push_base.
    # On le retire donc ici, ou toute la config est construite. Ce n'est pas un
    # detail cosmetique : push_by_setting_velocity ECRASE la vitesse de base
    # avec un tirage jusqu'a +/-0.4 m/s, soit plus que la commande maximale de
    # 0.3, et parfois a contresens. A l'entrainement l'episode dure 20 s et se
    # reinitialise ; en play episode_length_s vaut 1e9, donc ces poussees
    # tombent indefiniment sur le meme robot.
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

  # L'evenement de poussee s'appelle push_base, pas push_robot : le pop qui
  # etait ici ne retirait rien. Cette configuration veut bien un robot non
  # pousse, donc on retire le bon nom.
  cfg.events.pop("push_base", None)

  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0

  return cfg
