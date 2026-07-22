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
    # Torque feasibility is squeezed in late, after a gait exists: the
    # clipped execution is always feasible, so free exploration at full
    # scale first finds a dynamic, then this ramp squeezes the policy into
    # the torque-feasible action that already realizes it. Fine stages
    # (~1.3x, not 2x) give it room to actually converge at each level.
    cfg.curriculum["pd_demand_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "pd_demand_excess",
        "weight_stages": [
          {"step": 240_000, "weight": -0.5},
          {"step": 288_000, "weight": -0.75},
          {"step": 336_000, "weight": -1.0},
          {"step": 384_000, "weight": -1.5},
          {"step": 432_000, "weight": -2.0},
          {"step": 480_000, "weight": -2.5},
          {"step": 528_000, "weight": -3.0},
          {"step": 576_000, "weight": -4.0},
          {"step": 624_000, "weight": -6.0},
          {"step": 672_000, "weight": -8.0},
        ],
      },
    )
    cfg.curriculum["velocity_damper"] = CurriculumTermCfg(
      func=mdp.velocity_damper_progress,
      params={"start_step": 360_000, "end_step": 612_000},
    )
    cfg.curriculum["air_time_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "air_time",
        "weight_stages": [
          {"step": 144_000, "weight": 55.0},
          {"step": 216_000, "weight": 70.0},
          {"step": 288_000, "weight": 80.0},
        ],
      },
    )
    cfg.curriculum["min_foot_height_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "min_foot_height",
        "weight_stages": [
          {"step": 144_000, "weight": -32.0},
          {"step": 216_000, "weight": -41.0},
          {"step": 288_000, "weight": -50.0},
        ],
      },
    )
    # Contact-safety pressure grows in lockstep with air_time/min_foot_height
    # ambition, not lagging behind it.
    cfg.curriculum["impact_vel_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "impact_vel",
        "weight_stages": [
          {"step": 144_000, "weight": -1.0},
          {"step": 216_000, "weight": -1.3},
          {"step": 288_000, "weight": -1.6},
        ],
      },
    )
    cfg.curriculum["flat_support_weight"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "flat_support",
        "weight_stages": [
          {"step": 144_000, "weight": -11.0},
          {"step": 216_000, "weight": -13.0},
          {"step": 288_000, "weight": -15.0},
        ],
      },
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
  # Wide kernel: a tight one punishes the COM oscillation a long stride
  # naturally causes, implicitly biasing toward short quick steps.
  cfg.rewards["track_linear_velocity"].params["std"] = 0.30
  cfg.rewards["track_angular_velocity"].weight = 3.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.45

  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["pose"].params["command_name"] = "twist"
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["pose"].params["running_threshold"] = 1.5

  cfg.rewards.pop("soft_landing", None)
  cfg.rewards["impact_vel"] = RewardTermCfg(
    func=mdp.impact_velocity,
    weight=-0.7,
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
  # Penalizes the smoothed unclamped PD demand beyond the real effort
  # limits, so the policy can't lean on training-actuator saturation --
  # mc_mujoco and the real drives don't clamp like it does. Weight ramps in
  # late via the curriculum above (see rationale there).
  cfg.rewards["pd_demand_excess"] = RewardTermCfg(
    func=mdp.pd_demand_excess,
    weight=-1e-6,  # nonzero so Metrics/pd_demand_ratio logs from step 0
    params={
      "soft_ratio": 1.0,  # no margin above the real effort limit
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
    weight=-9.0,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "required_contacts_per_foot": 4,
    },
  )
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
  # Pose is a gentle pull toward a neutral, safe posture, not a gait-shape
  # constraint -- loosened across the whole body. ANKLE_R stays tighter:
  # lateral ankle stability is the one axis where "safe" and "loose" trade
  # off most directly.
  cfg.rewards["pose"].params["std_walking"] = {
    r".*CROTCH_P.*": 1.0,
    r".*CROTCH_R.*": 0.65,
    r".*CROTCH_Y.*": 0.65,
    r".*KNEE.*": 1.1,
    r".*ANKLE_P.*": 0.8,
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
  cfg.rewards["joint_torques_l2"].weight = -1e-5
  cfg.rewards["ankle_roll_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2,
    weight=-2e-3,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "actuator_pattern": r"^[LR]_ANKLE_R$",
    },
  )
  # Sustained plantarflexion torque is the signature of tiptoeing; kept well
  # below the roll weight since ankle pitch legitimately works at push-off.
  cfg.rewards["ankle_pitch_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2,
    weight=-2e-4,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "actuator_pattern": r"^[LR]_ANKLE_P$",
    },
  )
  # Dense (potential-based) shaping: every step during the swing pays
  # dPhi/dt instead of a lump sum at touchdown. See split_feet_air_time_dense.
  cfg.rewards["air_time"].func = mdp.split_feet_air_time_dense
  cfg.rewards["air_time"].weight = 40.0

  # Height shaping lives only in the min_foot_height floor below; per-step
  # clearance/swing-height taxes made short fast shuffling optimal.
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_swing_height", None)
  # Charged once per landing, not per airborne step: air time is free, only
  # landing with a low swing peak costs.
  cfg.rewards["min_foot_height"] = RewardTermCfg(
    func=mdp.split_feet_min_swing_height,
    weight=-25.0,
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
  # Smoothness pressure lives in joint (physical) space; action-space
  # rate/acc terms tax exploration noise itself and are kept small to avoid
  # premature std collapse. They use an L2-squared kernel on raw actions
  # (physical_rate = scale * raw_rate), so their weight must move with
  # scale^2 to preserve physical-space enforcement across a leg-scale
  # change -- these are calibrated for scale=5.0, rescaled by (1.5/5.0)^2.
  cfg.rewards["action_rate_l2"].weight = -0.0054
  # Split by phase below instead: stance/upper-body joints have different
  # smoothness needs than one blanket term captures.
  cfg.rewards.pop("action_acc_l2", None)
  cfg.rewards["stance_action_acc_l2"] = RewardTermCfg(
    func=mdp.stance_action_acc_l2,
    weight=-0.0162,
    params={
      "sensor_name": feet_ground_split_cfg.name,
      "left_joint_indices": list(range(6)),
      "right_joint_indices": list(range(8, 14)),
    },
  )
  cfg.rewards["upper_body_action_acc_l2"] = RewardTermCfg(
    func=mdp.joints_action_acc_l2,
    weight=-0.0162,
    params={"joint_indices": [6, 7, 14, 15, *range(16, 30)]},
  )
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
  cfg.rewards["air_time"].params["overflow_threshold"] = 0.8
  cfg.rewards["air_time"].params["overflow_weight_ratio"] = 0.1
  cfg.rewards["air_time"].params["command_name"] = "twist"
  cfg.rewards["air_time"].params["command_threshold"] = 0.1
  # Quadratic bonus + flat touchdown fee: reward rate grows with absolute
  # air time; steps shorter than threshold_max*sqrt(touchdown_cost) are net
  # negative.
  cfg.rewards["air_time"].params["power"] = 2.0
  cfg.rewards["air_time"].params["touchdown_cost"] = 0.30
  if cfg.curriculum is not None:
    # Each stage's break-even (threshold_max * sqrt(touchdown_cost)) stays
    # near the trailing operating point, so raising the ceiling never makes
    # the current gait suddenly unprofitable. All stages land before the
    # pd_demand torque ramp starts (step 240_000).
    cfg.curriculum["air_time_target"] = CurriculumTermCfg(
      func=mdp.air_time_target_curriculum,
      params={
        "reward_name": "air_time",
        "stages": [
          {"step": 72_000, "threshold_max": 0.6, "touchdown_cost": 0.25,
           "overflow_threshold": 0.95},
          {"step": 144_000, "threshold_max": 0.7, "touchdown_cost": 0.20,
           "overflow_threshold": 1.1},
          {"step": 216_000, "threshold_max": 0.85, "touchdown_cost": 0.135,
           "overflow_threshold": 1.3},
        ],
      },
    )
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

  cfg.curriculum["standing_envs"] = CurriculumTermCfg(
    func=mdp.standing_envs_curriculum,
    params={
      "command_name": "twist",
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
