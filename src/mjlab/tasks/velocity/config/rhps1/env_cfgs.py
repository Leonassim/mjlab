"""RHPS1 velocity environment configurations."""

from mjlab.asset_zoo.robots import RHPS1_ACTION_SCALE, get_rhps1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
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
  # Force-based counting (fields force + history) so the forceless proximity
  # contacts created by the leg-geom collision gap don't register as
  # collisions — only actual contact forces do.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="BODY", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="BODY", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=2,
  )
  # Hull-clearance sensors mirroring the deployment QP's minimalSelfCollisions
  # pairs, readable up to the collision gap set on these geoms (forceless
  # proximity contacts). One sensor per QP sDist group; thresholds live in the
  # matching leg_proximity_cost reward terms (QP sDist + 1 cm buffer).
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
  # Dedicated knee pair with a higher threshold: the mc_rtc knee hulls are
  # ~1.5 cm fatter than the mujoco meshes (measured 0.49 cm sch when the
  # sensor read 2.0 cm during lateral walking), so 3.5 cm here ~= 2 cm sch,
  # outside the QP knee damper zone (iDist 0.02, sDist 0.01).
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
    proj_grav_term.noise = Unoise(n_min=-0.1, n_max=0.1)
  for term in old_terms.values():
    if getattr(term, "history_length", None) is not None and term.history_length > 1:
      term.history_length = history_len
  if "command" in old_terms:
    old_terms["command"].history_length = history_len
    old_terms["command"].flatten_history_dim = True
  new_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      history_length=history_len,
      flatten_history_dim=True,
    )
  }
  new_terms.update(old_terms)
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
    cfg.observations["actor_history"].terms = ah_new

  if "critic" in cfg.observations:
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
      func=mdp.base_lin_vel
    )
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
      func=mdp.base_ang_vel
    )

  for group_name in ("actor_history", "critic", "teacher", "privileged"):
    if group_name in cfg.observations:
      terms = cfg.observations[group_name].terms
      terms.pop("phase", None)
      terms.pop("base_height", None)
      terms.pop("joint_acc", None)
      terms.pop("foot_height", None)

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
  twist_cmd.viz.z_offset = 1.0
  twist_cmd.ranges.lin_vel_x = (-0.1, 0.1)
  twist_cmd.ranges.lin_vel_y = (-0.15, 0.15)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.02)
  if cfg.curriculum is not None:
    cfg.curriculum["pd_demand_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "pd_demand_excess",
        "weight_stages": [
          {"step": 240_000, "weight": -1.0},
          {"step": 300_000, "weight": -2.0},
          {"step": 360_000, "weight": -3.0},
          {"step": 420_000, "weight": -4.0},
          {"step": 480_000, "weight": -5.0},
        ],
      },
    )
    cfg.curriculum["velocity_damper"] = CurriculumTermCfg(
      func=mdp.velocity_damper_progress,
      params={"start_step": 360_000, "end_step": 612_000},
    )
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

  site_names = ("left_foot", "right_foot")
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    if reward_name in cfg.rewards and "asset_cfg" in cfg.rewards[reward_name].params:
      cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["track_linear_velocity"].weight = 3.5
  cfg.rewards["track_linear_velocity"].params["std"] = 0.20
  cfg.rewards["track_angular_velocity"].weight = 3.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.35

  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["pose"].params["command_name"] = "twist"
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["pose"].params["running_threshold"] = 1.5

  cfg.rewards.pop("soft_landing", None)
  cfg.rewards["impact_vel"] = RewardTermCfg(
    func=mdp.impact_velocity,
    weight=-0.5,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      # 0.15 (was 0.10): landing_vel plateaued at the old soft limit across
      # runs and capped swing height/length — the impact penalty was the
      # physical ceiling on the air_time incentive.
      "limit": 0.15,
      "start_step": 0,
      "pre_contact_limit": 0.45,
      "pre_contact_window_s": 0.1,
      "always_limit": 1.2,
      "command_name": "twist",
      "always_command_threshold": 0.05,
    },
  )
  # One-leg-does-everything gaits are otherwise profitable (observed: left
  # foot 0.62 m/s marker speed vs right 0.28 on the 2026-07-13 run).
  cfg.rewards["air_time_symmetry"] = RewardTermCfg(
    func=mdp.feet_air_time_symmetry,
    weight=-1.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["no_double_flight"] = RewardTermCfg(
    func=mdp.no_double_flight_penalty,
    weight=-2.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.5,
    params={"sensor_name": self_collision_cfg.name},
  )
  # The deployment QP keeps the module's full minimalSelfCollisions (dampers
  # hard-stop at each pair's sDist on convex hulls). MuJoCo convexifies
  # collision meshes, so these sensors measure a comparable hull-hull
  # distance: each threshold = the pair group's QP sDist + 1 cm buffer, so the
  # policy stays out of the dampers' braking zone. Rest distances (measured):
  # thighs 2.9 cm, knees 6.7 cm, shoulder-chest 2.2 cm, arm-torso 6-20 cm —
  # all above their thresholds, no penalty in nominal posture.
  for prox_cfg, min_dist in (
    (leg_proximity_cfg, 0.02),  # QP sDist 0.01 (legs)
    (knee_proximity_cfg, 0.035),  # mc_rtc knee hulls ~1.5cm fatter than mujoco
    (arm_torso_proximity_cfg, 0.04),  # QP sDist 0.03 (elbow/wrist vs chest/body)
    (shoulder_chest_proximity_cfg, 0.01),  # QP sDist 0.001
    (shoulder_body_proximity_cfg, 0.04),  # QP sDist 0.03
    (wrist_thigh_proximity_cfg, 0.03),  # QP sDist 0.02
  ):
    cfg.rewards[prox_cfg.name] = RewardTermCfg(
      func=mdp.leg_proximity_cost,
      weight=-2.0,
      params={"sensor_name": prox_cfg.name, "min_dist": min_dist},
    )
  # Very strong penalty on the smoothed unclamped PD demand beyond the real
  # effort limits (Leo, 2026-07-17): policies must not lean on actuator
  # saturation -- mc_mujoco (no forcerange, by design) and possibly the real
  # drives do not clamp like the training actuator does. EMA-filtered so it
  # shapes the mean policy, not the exploration noise.
  # Weight 0 at start, ramped in by curriculum (below): the clipped
  # execution is always feasible, so a feasible reference trajectory exists
  # by construction -- the late ramp asks the policy to distill the clamp
  # into its own references once the gait exists, instead of taxing the
  # high-exploration phase into timidity (the air-time curriculum trap, in
  # reverse).
  # -1e-6 (not 0.0): zero-weight terms are skipped by the manager entirely,
  # and we want Metrics/pd_demand_ratio logged from step 0 to calibrate.
  cfg.rewards["pd_demand_excess"] = RewardTermCfg(
    func=mdp.pd_demand_excess,
    weight=-1e-6,
    params={
      # 1.2: an MPC-smooth reference (consistent q*/v*) keeps the servo
      # error at disturbance/kp, i.e. demand ratio ~1 -- the robot's MPC
      # controllers prove a dynamic gait can hold it. The margin tolerates
      # transients without allowing deep saturation reliance.
      "soft_ratio": 1.2,
      "cap": 1.0,
      "ema_dt": 0.04,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["torque_limit_margin"] = RewardTermCfg(
    func=mdp.joint_torque_limit_margin_penalty,
    weight=-0.08,
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
    weight=-5.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "required_contacts_per_foot": 4,
    },
  )
  # -12 (was -4): at zero command the policy stood on one foot ~100% of the
  # time (rate 0.10 with only 10% standing envs) and -4/s did not dislodge it.
  cfg.rewards["standing_single_support"] = RewardTermCfg(
    func=mdp.standing_single_support_penalty,
    weight=-12.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
    func=mdp.joint_torque_rate_l2,
    weight=-4e-5,
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
  cfg.rewards["pose"].params["std_walking"] = {
    r".*CROTCH_P.*": 0.85,
    r".*CROTCH_R.*": 0.45,
    r".*CROTCH_Y.*": 0.45,
    r".*KNEE.*": 0.95,
    r".*ANKLE_P.*": 0.6,
    r".*ANKLE_R.*": 0.05,
    r".*CHEST.*": 0.18,
    r".*SHOULDER_P.*": 0.08,
    r".*SHOULDER_R.*": 0.08,
    r".*SHOULDER_Y.*": 0.06,
    r".*ELBOW.*": 0.08,
    r".*WRIST.*": 0.05,
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
  cfg.rewards["angular_momentum"].weight = -0.2
  cfg.rewards["angular_momentum"].params["sensor_name"] = "robot/root_angmom"
  cfg.rewards["dof_pos_limits"].weight = -1.0
  cfg.rewards["joint_torques_l2"].weight = -1e-5
  cfg.rewards["ankle_roll_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2,
    weight=-2e-3,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "actuator_pattern": r"^[LR]_ANKLE_R$",
    },
  )
  # Light pitch-effort penalty: sustained plantarflexion torque is the
  # signature of tiptoeing. Kept well below the roll weight since ankle pitch
  # legitimately works during push-off.
  cfg.rewards["ankle_pitch_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2,
    weight=-2e-4,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "actuator_pattern": r"^[LR]_ANKLE_P$",
    },
  )
  cfg.rewards["air_time"].func = mdp.split_feet_air_time
  cfg.rewards["air_time"].weight = 20.0

  # foot_clearance (|z - target| x foot speed, every step) acts as a
  # per-meter tax on swinging while the feet are low: combined with the
  # per-airborne-step min-height penalty it made short fast shuffling optimal.
  # The only height shaping left is the min_foot_height safety floor.
  cfg.rewards.pop("foot_clearance", None)
  # foot_swing_height targeted exactly 0.15 m (two-sided quadratic), also
  # penalizing high steps. Foot height is only floored now: the air_time
  # incentive lets the gait pick its own natural height/stride above it.
  cfg.rewards.pop("foot_swing_height", None)
  # Charged once per landing (clamp(1 - peak/min_height, 0)), not per airborne
  # step: air time itself is free, only landing with a low swing peak costs.
  cfg.rewards["min_foot_height"] = RewardTermCfg(
    func=mdp.split_feet_min_swing_height,
    weight=-15.0,
    params={
      "min_height": 0.08,
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  cfg.rewards["foot_slip"].func = mdp.split_feet_slip
  cfg.rewards["foot_slip"].weight = -0.3
  # Smoothness pressure lives in JOINT space (physical, scale-independent):
  # action-space rate/acc penalties tax the exploration noise itself, which
  # drives premature std collapse (observed 2.0 -> 0.29 by iter 2300). They
  # are kept small; joint_acc_l2 below carries the anti-vibration signal.
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["action_acc_l2"].weight = 0.0
  cfg.rewards["stance_action_acc_l2"] = RewardTermCfg(
    func=mdp.stance_action_acc_l2,
    weight=-0.15,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "left_joint_indices": list(range(6)),
      "right_joint_indices": list(range(8, 14)),
    },
  )
  cfg.rewards["upper_body_action_acc_l2"] = RewardTermCfg(
    func=mdp.joints_action_acc_l2,
    weight=-0.15,
    params={"joint_indices": [6, 7, 14, 15, *range(16, 30)]},
  )
  # Physical anti-vibration term. Calibrated on the 2026-07-14 checkpoint
  # (dithering gait: sum-sq leg acc ~9e3 -> ~0.45/s; a smooth gait ~0.15/s).
  cfg.rewards["leg_joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-1e-4,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(r".*CROTCH.*", r".*KNEE.*", r".*ANKLE.*")
      )
    },
  )
  cfg.rewards["air_time"].params["sensor_name"] = feet_ground_split_cfg.name
  cfg.rewards["air_time"].params["threshold_min"] = 0.01
  cfg.rewards["air_time"].params["threshold_max"] = 0.5
  # 0.8 (was 2.0): the bonus saturates at threshold_max=0.5, so [0.5, 2.0]
  # was a dead zone where hovering on one foot dodged the touchdown fee for
  # free — the wide-exploration run converged to second-long low hovers with
  # collapsed tracking. Anything beyond 0.8 s now pays every step.
  cfg.rewards["air_time"].params["overflow_threshold"] = 0.8
  # 20 * 0.4 = 8: the anti-hover guard keeps the exact pre-boost scale;
  # only the (dt-diluted) landing bonus is amplified.
  cfg.rewards["air_time"].params["overflow_weight_ratio"] = 0.4
  cfg.rewards["air_time"].params["command_name"] = "twist"
  cfg.rewards["air_time"].params["command_threshold"] = 0.1
  # Quadratic bonus + flat touchdown fee: reward rate grows with absolute air
  # time; steps shorter than threshold_max*sqrt(touchdown_cost) are net
  # negative. The 2026-07-10 run converged exactly at the 0.15 break-even
  # (air_time_mean 0.174 s vs break-even 0.19 s): 0.30 moves the break-even to
  # ~0.27 s so short shuffling steps stay clearly unprofitable.
  cfg.rewards["air_time"].params["power"] = 2.0
  cfg.rewards["air_time"].params["touchdown_cost"] = 0.30
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
    cfg.events["foot_friction"].params["ranges"] = (0.5, 0.9)
  cfg.events.pop("push_robot", None)
  assert cfg.curriculum is not None

  # No air_time curriculum: the old stage-0 threshold_max=0.1 made every step
  # longer than 0.1s pay the same net bonus, so the high-exploration phase
  # locked the policy into high-frequency short steps before the later stages
  # could create a gradient toward long strides (observed on the 2026-07-13
  # run: air_time_mean peaked at 0.175 then fell to 0.137 at the stage
  # switch). Full threshold and weight from step 0 instead.

  cfg.curriculum["standing_envs"] = CurriculumTermCfg(
    func=mdp.standing_envs_curriculum,
    params={
      "command_name": "twist",
      # More standing practice (was 0.2 -> 0.1): with only 10% standing envs
      # the standing behavior was under-trained and its penalties barely
      # weighed in the batch return.
      "stages": [
        {"step": 0, "value": 0.3},
        {"step": 500 * 48, "value": 0.2},
      ],
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations[actor_group_name].enable_corruption = False
    if "actor_history" in cfg.observations:
      cfg.observations["actor_history"].enable_corruption = False
    cfg.observations.pop("teacher", None)
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("push_robot", None)
    cfg.curriculum.pop("air_time", None)
    cfg.curriculum.pop("air_time_weight", None)
    cfg.curriculum.pop("action_scale", None)
    cfg.curriculum.pop("standing_envs", None)
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

  return cfg


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

  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("push_robot", None)

  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0

  return cfg
