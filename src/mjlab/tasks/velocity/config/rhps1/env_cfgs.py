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
  # Explicit gait clock (see mdp.gait_phase_tracking / rewards["gait_phase"]
  # below): the policy needs to perceive the prescribed swing/stance timing
  # to act on it, not just be rewarded for matching it by chance.
  # reward_name passed explicitly (not left to the function default) so
  # play.py --fast can see that this observation depends on the gait_phase
  # reward term: the clock lives in that term, and fast mode drops any reward
  # nothing references. A default argument is invisible to that scan.
  new_terms["gait_phase"] = ObservationTermCfg(
    func=mdp.gait_phase_obs, params={"reward_name": "gait_phase"}
  )
  # Feed back the action as *executed*, not as requested. The actuator projects
  # any command whose PD demand exceeds the effort limit back onto the
  # executable set, so on those steps mdp.last_action reports an intent the
  # plant never carried out, and the policy's own history misdescribes the
  # dynamics it is trying to model (18% of steps early in the 2026-07-29 run).
  #
  # This is not about torque saturation, which is normal for high-gain position
  # control and costs nothing -- kp=20000 against a 140 N.m hip has a 0.4 deg
  # linear range, the real robot rides its limit constantly. It is about two
  # different commands producing one identical execution: the policy cannot
  # notice that if it only ever observes what it asked for.
  #
  # Applied to the actor group and to actor_history below; they must agree, or
  # the history says something different from the current step about the same
  # quantity.
  # executed_action again: the projection is back on for the combination run
  # (_TORQUE_FEASIBILITY_RATIO = 1.0), so the raw action and the executed one
  # diverge on every step where the projection bites -- which is most of them,
  # the feasible half-window is 0.143 action units against a 1.0-unit scale.
  # Feeding back intent instead of execution would put the hidden state straight
  # back in, and this pairs with the history below rather than replacing it.
  #
  # history_length 5 on actions (Leo, 2026-08-01), treated as a base change and
  # not as an ablation. Rationale: with the velocity-target EMA at alpha=0.8,
  # qd* depends on the whole history of position targets and nothing in the
  # observation carried it -- the actuator was non-Markovian from the policy's
  # point of view. Five frames recover 1 - 0.8^5 = 67% of that state. Costs 120
  # observation dimensions.
  if "actions" in new_terms:
    new_terms["actions"] = ObservationTermCfg(
      func=mdp.executed_action, history_length=history_len, flatten_history_dim=True
    )
  # gait_phase history: asked for in the same breath. Less defensible on its own
  # -- the phase is a deterministic clock, so the current frame already
  # determines every past one -- but it is 16 dimensions and it makes the two
  # history-bearing term families consistent.
  if "gait_phase" in new_terms:
    new_terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase_obs,
      params={"reward_name": "gait_phase"},
      history_length=history_len,
      flatten_history_dim=True,
    )
  # Peak raw torque of the last policy step, per joint, normalised by the effort
  # limit (Leo, 2026-08-04). The critic already had joint_torques, but post-clamp:
  # past the limit every command reports the same number, so it cannot say how far
  # past. The actor had nothing and inferred saturation from joint_pos/joint_vel.
  #
  # A longer window than history_len (5), because that is the point *here* -- a
  # saturation builds over tens of milliseconds and the policy needs to see it
  # coming -- while on command, which changes on a 3-8 s resample, it would be
  # dimensions of nothing. Kept local for that reason: at 20 across every block
  # the observation goes 266 -> 1466 dims; local, it is 566, and the C++ side has
  # one new block to build instead of five to resize.
  #
  # 20, and local rather than raising history_len for every block: a saturation
  # builds over tens of milliseconds and the policy needs to see it coming, while
  # on command -- which changes on a 3-8 s resample -- the same window would be 45
  # dimensions of nothing. At history_len 20 everywhere the observation would go
  # 266 -> 1466 dims and the C++ deployment would have five blocks to resize
  # instead of one to add. Local, it is 866.
  #
  # 10 (566-dim actor obs), not the 20 first asked for. Measured grid on an 11.3 GB
  # card, all failures being "Warp CUDA error 2: out of memory" in
  # wp.capture_launch at iteration 0:
  #
  #   history  video  expandable_segments   result
  #      20     yes          no             OOM
  #      20     no           no             OOM
  #      10     yes          no             OOM
  #      10     no           no             OK  (10.16 GB)
  #       5     yes          yes            OK  (10.31 GB)
  #      20     yes          yes            OOM
  #      10     yes          yes            OK  (10.66 GB)   <- this
  #
  # Two lessons. Most of those failures were fragmentation, not capacity: running
  # the trainer under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is what
  # makes video and a 566-dim observation coexist, and it must be tried before
  # shrinking anything -- four restarts were spent cutting history and dropping
  # video to work around an allocator setting. But it is not unlimited: at 866 dims
  # the run OOMs even with it, so 20 is genuinely out of reach here.
  #
  # THE TRAINER MUST BE LAUNCHED WITH THAT ENV VAR. Without it this config OOMs.
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
  # vel_ramp_rate left at None (teleporting command) -- UNTESTED, not rejected.
  #
  # Setting it to 0.5 to match NewRLQPController's ramp would put the walk-to-stand
  # transition, the regime the deployed robot spends ~0.6 s in on every stop, into
  # the training distribution; today it is never sampled at all, because the command
  # teleports at every resample. That argument still stands.
  #
  # Two runs on 2026-08-03 appeared to reject it. They are void: both trained with
  # `num_envs` at its default of 1 instead of 4096 (the launch command omitted
  # --env.scene.num-envs), so PPO saw 48 samples per iteration instead of 196608 and
  # no configuration could have learned. Check the "Number of environments" line the
  # trainer prints at startup before trusting any comparison.
  #
  # The one thing measured off-run and still valid: scratch/cmd_band.py shows the
  # ramp does NOT starve gait_phase -- it leaves slightly more time above the 0.1
  # gate, not less.
  twist_cmd.viz.z_offset = 1.0
  twist_cmd.ranges.lin_vel_x = (-0.1, 0.1)
  twist_cmd.ranges.lin_vel_y = (-0.15, 0.15)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.02)
  if cfg.curriculum is not None:
    # velocity_damper disabled for now (2026-07-26). It ramps the actuator
    # safety projection that mirrors the mc_rtc QP KinematicsConstraint, which
    # matters for deployment -- but it only starts at step 360_000 (iter 7500)
    # and no run has ever survived that far, so it has literally never
    # executed. Re-enabling it belongs in a pass of its own, once a run
    # actually reaches that far and the many reward changes made today have
    # been evaluated on their own. FiniteDifferencePdActuator defaults
    # velocity_damper_progress to 0.0, and the projection is an explicit
    # no-op at 0, so leaving the curriculum out simply keeps it inactive.
    # No air_time_weight curriculum (removed 2026-07-25, with the
    # air_time_target one). It used to ramp 40 -> 80 to demand progressively
    # longer swings; that job now belongs entirely to the gait clock, which
    # prescribes a 0.4s swing at every speed. air_time keeps a fixed
    # threshold_max of 0.5s, so escalating its weight would only sharpen a
    # pull toward 0.5s *against* the clock's 0.4s -- the opposite of what
    # the ramp was for. Its remaining job (require a genuine lift-off, not
    # a light-footed drag; see the air_time block further down) is a
    # constant-strength guard, not something that needs to grow.
    #
    # Historical note, since this was one of the four curricula involved:
    # air_time_weight, impact_vel_weight, flat_support_weight and
    # air_time_target all used to jump together on step 216_000 (iter
    # ~4500), which compounded into a full training collapse -- fell_down
    # from a noisy ~0.15-0.25 baseline to >6/episode over the following
    # ~5000 iterations with no recovery. The survivors are now staggered;
    # see min_foot_height_weight below.
    # Metrics/peak_height_mean plateaued at ~0.006-0.009m (target 0.08m, net
    # ~0.055m after the ~0.023m ankle-site offset -- see 2026-07-24 visual
    # check) through the entire -25 -> -41 range with zero improvement. Two
    # fixes together: pushed the weight ceiling much higher (-200), and
    # curriculum the min_height target itself instead of fixing it at the
    # final 0.08m from step 0 -- 0.08m was simply out of reach the whole
    # time, so no weight was going to close that gap on its own (same
    # break-even idea as air_time_target_curriculum: each stage's target
    # should be reachable from the previous stage's converged behavior).
    # Fine ~1.3x weight stages (not the old 3-stage jumps that only reached
    # -50), same lesson as pd_demand_excess's regrade: many small steps let
    # the policy actually converge at each level, few big ones risk
    # yanking it past what it can absorb. Steps deliberately offset from
    # 144k/216k/288k (the air_time/impact_vel/flat_support jump points) to
    # avoid stacking multiple curricula's hardest jump on the same step,
    # the likely cause of the iter-4500 destabilization.
    # min_foot_height_weight curriculum removed (2026-07-26). It ramped the
    # weight -25 -> -200 (and min_height 0.030 -> 0.080) across 8 stages. Two
    # runs' worth of evidence say it does not work and now actively hurts:
    #   - Metrics/peak_height_mean never responded to the weight anywhere in
    #     the -25 -> -200 range across earlier runs.
    #   - In run 2026-07-25_18-33-25 the first stage (weight -25 -> -33, at
    #     step 100_000 / iter 2083) produced a clean step change in
    #     Episode_Termination/fell_down, from ~0.26-0.30 before to
    #     0.37-0.54 immediately after, with 7 more such stages queued behind
    #     it.
    #   - Over that same window peak_height_mean went *down* (0.0081 ->
    #     0.0049 vs the previous run), i.e. the pressure rose while the
    #     behavior it targets got worse.
    # Since peak height (~0.006) sits far below even the first stage's 0.030
    # target, the one-sided deficit clamp(1 - peak/min_height, 0) is nearly
    # saturated regardless, so the ramp mostly rescales a constant penalty
    # rather than sharpening a gradient. Base weight -25 / min_height 0.030
    # stays; see the note on swing-height shaping in the air_time block for
    # why the foot skims rather than lifts.
    # impact_vel_weight curriculum removed (2026-07-27): base weight set
    # directly to -2.0 (safety-priority budget bump, see impact_vel
    # registration), already above every stage this used to ramp through
    # (-1.0 -> -1.3 -> -1.6). Left wired up, the first stage firing at step
    # 170k would have *downgraded* the weight back to -1.0. Same reasoning
    # as the flat_support_weight removal directly below.
    # flat_support_weight curriculum removed (2026-07-26), same reasoning as
    # air_time_target and min_foot_height_weight before it: it ramped -18 ->
    # -30 on a metric that had not moved across three runs (2.0 -> 2.5 of 4,
    # flat through a -9 -> -18 base change). Ramping was never going to fix a
    # term whose obstacle is a competing penalty, not insufficient pressure.
    #
    # Every curriculum that used to live in this block has now been removed
    # (see comments above) -- velocity_damper, air_time_weight,
    # min_foot_height_weight, impact_vel_weight, flat_support_weight.
    #
    # No torque_feasibility curriculum either, and this one is worth stating
    # rather than leaving as an absence, because a ramp 3.0 -> 1.0 was drafted
    # here first (2026-07-29) and then dropped as pointless.
    #
    # The projection is set at ratio 1.0 constantly in rhps1_constants.py. At
    # 1.0 it delivers exactly the torque MuJoCo's effort clamp already
    # delivered -- tau is affine and increasing in q*, so clamping tau and
    # projecting q* onto tau's preimage are the same operation. Same dynamics,
    # same return, so there is no adaptation for a curriculum to ease: a ramp
    # would interpolate between two settings with *identical* physics, buying
    # nothing while re-admitting un-executable commands early on. That is the
    # opposite of the point.
    #
    # ``curriculums.torque_feasibility_progress`` still exists as an escape
    # hatch, unregistered. Wire it only if a run somehow argues for it.
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
  # Wide kernel: a tight one punishes the COM oscillation a long stride
  # naturally causes, implicitly biasing toward short quick steps. Widened
  # 0.30 -> 0.40 alongside the low-cadence gait clock (originally
  # stride_frequency_target, now gait_phase): a lower stride count means
  # for the same commanded speed, stride length (and the COM velocity
  # ripple within each stride) must grow -- give it more room here rather
  # than fight it with a tight kernel. Weight left untouched: this is a
  # tolerance-shape fix, not an importance/scale one, and changing both at
  # once would make either change hard to attribute in the next run.
  cfg.rewards["track_linear_velocity"].params["std"] = 0.40
  cfg.rewards["track_angular_velocity"].weight = 3.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.45

  # 0.5 -> 1.5, alongside the mean(exp) reform in mdp.variable_posture. This is
  # the only term that tells the robot *which way* to move to fix a bad
  # standing posture -- everything else about standing is a verdict, not a
  # direction -- and at 0.5 its entire achievable range was 4% of the
  # standing_single_support penalty it is meant to counterbalance.
  cfg.rewards["pose"].weight = 1.5
  cfg.rewards["pose"].params["command_name"] = "twist"
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["pose"].params["running_threshold"] = 1.5

  cfg.rewards.pop("soft_landing", None)
  cfg.rewards["impact_vel"] = RewardTermCfg(
    func=mdp.impact_velocity,
    # -2.0 (was -0.7, 2026-07-27): realized only -0.27 of a -17.1 budget --
    # essentially unpenalized despite being one of the four explicit safety
    # priorities (low impact velocity, don't break knees). Cheap headroom,
    # same budget-ratio reasoning as flat_support above. Skips the existing
    # impact_vel_weight curriculum's slow ramp to -1.6 by iter ~4500; that
    # ramp was ambition-paced (matched to how fast air_time/flat_support grew
    # harder to reach), not safety-paced, and this term does not need to
    # wait on it.
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
  # Video review (run 2026-07-23_16-09-31) showed the swing foot kicking
  # backward before swinging forward -- air_time only rewards *duration*
  # aloft, not direction, so a longer, looping swing banks the same credit
  # as a direct one. Targeting knee flexion directly was considered and
  # rejected: real swing is hip-driven (double-pendulum model), the knee
  # bends passively as the shank trails the hip-accelerated thigh, so a
  # foot/knee that moves backward purely from correct hip-driven flexion is
  # legitimate and would be wrongly punished by constraining knee or foot
  # motion directly. Constrain the hip's direction instead and let
  # knee/foot follow.
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
  # Periodic Reward Composition (Siekmann et al., Cassie): an explicit gait
  # clock prescribes swing/stance timing directly (paired with
  # mdp.gait_phase_obs so the policy can perceive and act on it), rather
  # than the reactive/post-hoc approach the rest of this file uses --
  # air_time/min_foot_height/stride_frequency_target all stalled for
  # thousands of iterations despite escalating weights, since they only
  # ever reward a gait the policy already stumbled onto, not steer it
  # there.
  #
  # Cadence is commanded-speed dependent (period_slow -> period_fast over
  # command_threshold -> command_ref), NOT curriculum-driven. A curriculum
  # link to air_time_target's threshold_max was tried 2026-07-25 and
  # reverted the same day: run 2026-07-25_18-33-25 was stable (fell_down
  # noisy but bounded, ~0.03-0.3) right up through that curriculum's first
  # stage at step 72_000/iter 1500, then entered a slow, continuous,
  # never-recovering climb starting ~iter 2100 that compounded into full
  # collapse (fell_down >70/episode) by iter ~3200. Speed-dependence is a
  # different mechanism and does not reintroduce that failure: it varies
  # smoothly and per-env with a quantity the policy already observes (the
  # command), the way ``amplitude`` does, rather than snapping the cadence
  # for all 4096 envs at once at a curriculum step boundary.
  #
  # Targets: 1.0s per step at the slow end (period 2.0s), ~0.55s per step
  # at command_ref and above (period 1.1s) -- a step is a half-cycle. Fixed
  # cadence was the structural reason low-speed commands could only ever
  # produce tiny steps (step length = v*T/2), the "petits pas" problem this
  # whole line of work has been chasing.
  #
  # swing_duration is a duration, not a ratio, so single-support time stays
  # ~0.4s at every speed and the extra cycle time at low speed goes into
  # double support (= 1 - 2*swing_ratio) instead of into a 1.0s one-legged
  # balance the robot cannot hold. See mdp.gait_phase_tracking's docstring.
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
  # Logging-only: kept at a near-zero weight purely so
  # Metrics/pd_demand_ratio_mean/max still track hardware-readiness. The two
  # curriculum ramps that used this as a training signal (up to -8.0) both
  # produced the same failure mode -- large enough to matter, it also made
  # the policy timid on the big corrective actions needed to catch a
  # stumble, so falls became more frequent, and a fall's own chaotic demand
  # fed the same penalty right back in. The actual torque-feasibility signal
  # is now mdp.pd_action_guidance_target, consumed as a direct actor-space
  # auxiliary loss (rl_ext.TorqueGuidedPPO) instead of a reward: it doesn't
  # route through the return/advantage, so it can't create this loop.
  cfg.rewards["pd_demand_excess"] = RewardTermCfg(
    func=mdp.pd_demand_excess,
    # -1e-6 -> -1e-4 (2026-07-26). Enabled deliberately, at a weight small
    # enough that it cannot dominate a recovery. The problem it addresses is
    # real and measured on every run: Metrics/pd_demand_ratio_mean climbs to
    # 3-4, i.e. the policy asks for three to four times the torque the
    # actuator can deliver. Training never notices because the simulated
    # actuator clamps silently, but hardware that clamps differently is
    # exactly how mc_mujoco blew up on 287 N.m demanded from a 35 N.m hip.
    # Kept deliberately small: at a large weight this term caused two
    # collapses earlier in the session by making the policy timid about the
    # big corrective actions that catch a stumble -- suppressing the very
    # recoveries that prevent falls, whose own chaotic demand then fed the
    # penalty back in. A constant small weight gives a gradient toward
    # feasibility without ever outbidding a recovery, and is more legible
    # than the late curriculum the original comment suggested.
    # -1e-3 (was -1e-4, 2026-07-28). At -1e-4 the realized value was -0.001,
    # i.e. 0.01% of the penalty budget -- inert, while
    # Metrics/pd_demand_ratio_mean sits at 5.6 (max 350): the policy asks for
    # ~5.6x the torque the actuator can deliver and nothing pushes back.
    # Still deliberately small: the history above is real, a large weight
    # here made the policy timid on stumble recoveries twice. The heavy
    # lifting is meant to come from torque_guidance_coef (rl_cfg.py), which
    # bypasses the advantage entirely and therefore cannot form that loop.
    # -0.05 (was -1e-3, 2026-07-29), and re-banded. Its job changed: the hard
    # cap is now structural (FiniteDifferencePdActuator's
    # torque_feasibility_ratio = 3.0 projects the position target, so the demand
    # cannot exceed 3x the limit by construction), and this term only has to
    # supply a *gradient before the wall* so the policy prefers the interior
    # rather than riding the projection -- riding it would recreate exactly the
    # flat plateau the projection exists to remove, just at ratio 3 instead of
    # infinity.
    #
    # That is why the two earlier collapses at large weights (up to -8.0, both
    # producing a policy too timid to catch a stumble, whose falls then fed the
    # penalty back in) do not argue against -0.05 here: with the projection in
    # place a big corrective action is still *executed*, merely projected onto
    # the feasible set, so the recovery is not suppressed -- only its infeasible
    # component is. Budget check: excess is capped at 1.5 per joint and
    # realistically only the leg joints pay, so the realized term should land
    # around -0.15 to -0.3, a few percent of the penalty budget, nowhere near
    # the ~2.5:1 TOTAL-/TOTAL+ working ceiling. Watch it on the first run.
    #
    # -0.02 (was -0.05 the same day), after measuring the live budget on run
    # 2026-07-29_01-13-36 at iter 9096 rather than estimating it. Realized
    # there was -0.0085 at weight -1e-3, i.e. sum(excess) ~ 8.6 at
    # soft_ratio 1.0 / cap 1.0; re-banded to 0.7 / 4.0 that sum lands around
    # 40-50, so -0.05 would have realized ~-2.0 -- third-largest penalty in the
    # run, and it would have pushed the torque family (this +
    # torque_limit_margin -0.57 + joint_torques_l2 -1.03 +
    # joint_torque_rate_l2 -0.75) to ~35% of a 10.3 penalty budget against an
    # 11.1 task budget. That concentration is how the two earlier collapses
    # started. -0.02 realizes ~-0.8, keeps the family near 28%, and is still
    # 20x the inert level it replaces.
    #
    # It does not need to be large: with the projection active, an infeasible
    # command no longer *buys* anything (it produces the same torque as its
    # projection), so this term only has to break a tie, not win an argument.
    # -0.02 and the 0.7/4.0 band, finally applied (2026-08-01): every comment
    # above described them, the code was still at -0.001 / 1.0 / 1.0. Same
    # comment-ahead-of-code gap as the velocity filter had.
    #
    # Caveat this run carries, stated because the reasoning above assumes
    # otherwise: the band and the weight were derived *with the projection
    # active* ("0.7, i.e. below the projection's ratio of 1.0", "with the
    # projection in place a big corrective action is still executed, merely
    # projected, so the recovery is not suppressed"). Here the projection is
    # OFF, so this term is alone against the two collapses documented above --
    # a policy too timid to catch a stumble, whose falls then feed the penalty
    # back in. Those were at weights up to -8.0, i.e. 400x this one, and the
    # realized value here should land near -0.8. Watch Episode_Termination/
    # fell_down against the realized Episode_Reward/pd_demand_excess: if falls
    # climb while the term grows, that is the loop restarting.
    weight=-0.02,
    params={
      # 0.7, i.e. *below* the projection's ratio of 1.0, not above it. This
      # term is the only thing that moves the raw action into the window: the
      # projection is local to the actuator's compute, so
      # ``data.joint_pos_target`` (what this reads) keeps the raw target and
      # the measured ratio stays honest -- but a projection supplies no
      # gradient of its own, it just discards the excess. A soft_ratio of 1.0
      # or more would only start paying once the command is already
      # infeasible, and riding the projection is itself a flat region; 0.7
      # gives a pull that begins inside the feasible set.
      #
      # cap 4.0, not 2.0: the cap is a clamp, so every joint past it pays the
      # same and the gradient there is exactly zero -- capping at 2.0 with the
      # live ratio at 3.26 (run 2026-07-29_01-13-36, iter 9096) would have put
      # most leg joints in a second flat region, which is the same mistake this
      # whole change is undoing. 4.0 keeps the observed range on the sloped
      # part while still bounding the tail (the ratio max has hit 350).
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
  # -18 (was -9, 2026-07-24): flat_support_contacts_mean plateaued at
  # ~1.97-1.99/4 for 5000+ iterations despite the curriculum already
  # reaching -13 (run 2026-07-23_16-09-31) -- doubled the base alongside the
  # curriculum stages below rather than trust the same range to work this
  # time.
  # -18 -> -25 (2026-07-26). 4/4 is geometrically reachable: at the nominal
  # keyframe all four contact boxes are exactly coplanar (world z = -0.0020
  # each), and they sit only 3 mm proud of the real ankle-link mesh, so they
  # are a faithful stand-in for the sole. Yet after a couple of hundred
  # settling steps under its own control the robot holds a tilted stance --
  # the same four boxes then span 6.5 mm in height and only two of them
  # report contact. So the deficit is a control failure, not a modelling
  # one, and the weight was never the blocker either (2.0 -> 2.5 of 4 across
  # three runs, flat through a -9 -> -18 base change). What was blocking it:
  # levelling a foot costs ankle torque, and ankle_roll + ankle_pitch were
  # taxed harder than this reward paid -- realized ~-4.8 against
  # flat_support's -4.2, so staying tilted was simply cheaper. Those two are
  # cut in the same commit; this now clearly outweighs them.
  cfg.rewards["flat_support"] = RewardTermCfg(
    func=mdp.flat_support_penalty,
    # -11 (was -7, then -15 before that, 2026-07-27): budget-ratio framework
    # replaces per-term isolation. TOTAL-/TOTAL+ realized ratio measured
    # ~2:1 at -7 (survived) vs ~4:1 at pre-rescale weights (collapsed) --
    # 2.5:1 is the working safety ceiling. flat_support's realized share was
    # only -2.94 of -17.1, cheap headroom to buy more of exactly what the
    # user's safety priorities call out (foot flat at every contact). The
    # escape closure it carries is unchanged.
    weight=-11.0,
    params={
      # Stays at 4: all four corners genuinely can touch at once. The four
      # contact boxes are exactly coplanar at the nominal keyframe (world
      # z = -0.0020 each), and the reason a settled robot only lands 1.5 of
      # them is a ~3.4 deg steady-state droop at the ankle roll joint -- a
      # proportional controller only produces the torque that holds the foot
      # level by sitting at some position error, and 3.4 deg across a 11 cm
      # sole lifts one edge by the 6.5 mm measured between corners. The policy
      # can cancel that by commanding an offset ankle target: 3.4 deg is 1.9
      # action units against a typical output of 2.4. It never did, because
      # ankle_roll_torque and ankle_pitch_torque taxed exactly that
      # compensation harder than this term paid for the result -- both are
      # removed in this same commit, so the correction is affordable for the
      # first time. If the metric still parks below ~3 after a full run, then
      # lower this; the loss/gain terms keep supplying gradient either way, so
      # an out-of-reach level no longer silences the whole term the way the
      # old absolute-only version did.
      "sensor_name": feet_ground_split_cfg.name,
      "command_name": "twist",
      "required_contacts_per_foot": 4,
      # Ablation 3 (2026-07-31). Counts a corner as down when it sits within
      # 1 mm of the *lowest* corner of the same foot, instead of waiting for
      # the contact sensor to fire. Relative z, not an absolute plane: the
      # four patches share one rigid body, so differences between their centre
      # heights equal differences between their lowest corners and tilt
      # cancels without needing the orientation. Comparing centres to the
      # ground does not work -- at a 0.0525 m half-length, 0.01 rad of tilt
      # moves the lowest corner by 525 um.
      #
      # 1 mm is deliberately small, and the sensitivity is the reason: it is
      # 0.44 deg across the 130 mm sole, against the 3.4 deg ankle droop this
      # term exists to remove. So it forgives sensor-threshold noise and the
      # irregularity the real robot will have, without forgiving a tilted
      # stance. Measured effect on a settled robot: 2.15 -> 3.87 corners.
      #
      # This replaces the geom margin+gap route, which is unavailable here:
      # mujoco_warp raises NotImplementedError for a non-zero margin on
      # box-box pairs under MULTICCD.
      "corner_tolerance": 0.001,
    },
  )
  # -12 -> -6 (2026-07-26). At -12 this had become the largest single term in
  # the objective (-5.44 realized, more than any positive reward) while
  # provably not changing the behaviour it targets: the policy holds one foot
  # 27 mm up for 99% of standing time, and four separate hypotheses for why --
  # broken measurement, unavoidable transitions, exploration noise, missing
  # gradient -- were each tested and refuted. A large penalty that does not
  # move behaviour does not steer the policy; it just injects variance into
  # the value function, which is what destabilised the run where this was
  # briefly doubled. Halved so it still expresses the preference without
  # dominating, pending an actual explanation. standing_joint_vel now covers
  # the "hold still" half of the intent with a dense signal.
  # Back to -12 (2026-07-26, same pass that closed flat_support's escape
  # hatch). Halving it was the wrong call: the term was not weak, it was being
  # cancelled -- lifting a foot saved 5.86 on flat_support against this
  # penalty's 6.0. With that saving now turned into a 9.14 *extra* charge, the
  # gap between one foot and two at zero command is ~21 per step instead of
  # 0.14, which is the large, unambiguous shift the behaviour needs.
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
  # -3e-5 (was -1e-5): plain L2 on realized (post-clamp) torque, no
  # threshold/margin -- just "less torque, everywhere, all the time". Unlike
  # pd_demand_excess (unclamped demand, routed through the return, caused
  # two separate collapses this session) this is smooth and monotonic, so a
  # 3x bump is a reasonable first non-marginal step rather than a risky one.
  # At -1e-5 it already contributed ~-0.87 per step (Episode_Reward, run
  # 2026-07-23_16-09-31) -- not actually negligible despite the tiny-looking
  # weight, since raw squared torques are large; retune further after a run.
  # One consolidated torque term instead of three overlapping ones (see
  # mdp.joint_torques_weighted_l2). Ankles carry the emphasis: weakest
  # actuators on the leg, and low ankle effort goes hand in hand with a
  # well-placed flat foot rather than against it. 4x on roll (45 N.m limit,
  # and the axis that decides whether the sole sits flat), 2x on pitch
  # (65 N.m, legitimately works at push-off). Far below the ~18x the old
  # separate ankle_roll_torque effectively applied, which made it the single
  # largest term in the whole objective.
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_weighted_l2,
    # -1.2e-5 (was -3e-5, 2026-07-27): uniform penalty rescale, see action_jerk.
    weight=-1.2e-5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      # ANKLE_R 4.0 -> 1.5 (2026-07-28). flat_support has been stuck at
      # ~2.67/4 contacts for thousands of iterations, and its own comment
      # explains why the policy never learns the fix: holding the sole flat
      # against the ~3.4 deg ankle-roll droop needs *sustained ankle-roll
      # torque*, and that comment assumed the conflict was gone once
      # ankle_roll_torque/ankle_pitch_torque were deleted. It wasn't -- this
      # 4x coefficient reproduces the same tax in reduced form, on exactly
      # the joint whose torque buys flat contact. Raising flat_support
      # instead would be pushing harder on an already-dominant term (-3.56,
      # the largest single penalty) against a cost it cannot outbid.
      "coeffs": {r"ANKLE_R": 1.5, r"ANKLE_P": 2.0},
    },
  )
  # ankle_roll_torque / ankle_pitch_torque removed (2026-07-26). Three terms
  # were taxing torque *usage* -- these two plus joint_torques_l2, which
  # already covers every joint including the ankles -- while only one,
  # torque_limit_margin, guards what actually protects hardware: exceeding the
  # actuator limit. The ankles were therefore double-taxed for doing necessary
  # work, and measurably so (realized -4.8 combined against flat_support's
  # -4.2). Since the ankle is the only joint that can level a sole, that was a
  # direct conflict with the flat-contact priority: staying tilted was cheaper
  # than correcting. Tuning them up and down across three runs never resolved
  # it, which is the signature of a term that should not exist rather than one
  # that is mis-weighted. joint_torques_l2 (-3e-5) and torque_limit_margin
  # (-0.08) remain; watch Metrics/torque_limit_ratio_mean for ankle saturation.
  # air_time retired (2026-07-26). It and gait_phase both governed swing
  # duration, which is one job: the clock prescribes it up front and the actor
  # can see it coming through mdp.gait_phase_obs, a channel air_time never
  # had. Same reasoning that retired stride_frequency_target earlier -- once an
  # explicit clock owns the timing, the post-hoc measurements of that timing
  # should stop steering it. Its realized contribution had already fallen to
  # roughly -0.1 to -0.6 per step, so little is being given up numerically.
  #
  # Caveat worth watching: air_time read the binary contact sensor and was
  # therefore the strictest test that a foot genuinely left the ground.
  # gait_phase scores swing as exp(-(F/force_std)^2), which a merely unloaded
  # foot partly satisfies -- though force_std was tightened 30 -> 12 in this
  # same pass, taking a 10 N skim from 89% of the reward down to 50%.
  # min_foot_height still requires a real airborne phase, since its landing
  # edge only fires after all four corners have cleared. If
  # Metrics/peak_height_mean and air_time_mean start diverging (long air time,
  # flat trajectory), a skimming gait is the thing to suspect.
  cfg.rewards.pop("air_time", None)
  # Height shaping lives only in the min_foot_height floor below; per-step
  # clearance/swing-height taxes made short fast shuffling optimal.
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_swing_height", None)
  # Charged once per landing, not per airborne step: air time is free, only
  # landing with a low swing peak costs.
  #
  # Landing-triggered -> clock-gated (2026-07-30). The comment above describes
  # the defect without naming it: "air time is free, only landing with a low
  # swing peak costs" also means *never leaving the ground costs nothing*. A
  # landing is the transition from fully airborne to any contact, so a robot
  # that keeps a toe down never triggers one and never pays.
  #
  # Measured by perturbation on model_450: multiplying the deterministic
  # policy's action by 3 made it move 7x faster (0.0072 -> 0.0545 m/s) with
  # zero terminations, and this term alone went -0.51 -> -12.64 -- half of the
  # total reward lost. At weight -100 and a 5 mm peak against a 15 mm floor,
  # every attempted step cost ~67 while standing still cost 0. The term whose
  # only purpose is to make the robot lift its foot was the largest single
  # penalty on lifting it.
  #
  # That also explains what env_cfgs recorded as an unexplained plateau:
  # peak_height_mean "never responded to the weight anywhere" from -25 to -200
  # across two curricula. It could not -- more weight makes attempting a step
  # more expensive, so each escalation deepened the trap.
  #
  # mdp.clock_swing_height_deficit charges the shortfall over the *prescribed*
  # swing window instead, whether or not the foot actually left the ground:
  # standing through a swing pays full, 5 mm pays a little less, 15 mm pays
  # nothing. Monotone in height, impossible to dodge by staying planted, and no
  # per-landing bonus to farm with a chopped cadence.
  #
  # -100 -> -5: it now fires on ~25% of steps per foot instead of ~0.05
  # landings per step, roughly a 10x higher rate. -5 puts a fully planted robot
  # at ~2.4 per step from this term, comparable to flat_support, rather than
  # crushing everything else.
  cfg.rewards["min_foot_height"] = RewardTermCfg(
    func=mdp.clock_swing_height_deficit,
    weight=-5.0,
    params={
      # In TRUE sole clearance since the left/right_foot sites moved to the
      # sole plane (rhps1_constants.py, 2026-07-26). The old 0.030 was read
      # against a site 20 mm up the ankle, so it only ever asked for 10 mm of
      # real clearance -- and the robot delivered 2-4 mm because the reward
      # was mismeasuring landings anyway. 0.015 is a genuine 15 mm floor.
      "min_height": 0.015,
      # sensor_name / command_name / command_threshold are gone: the window is
      # the gait clock's, and the clock's own amplitude does the command gating.
      "reward_name": "gait_phase",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  cfg.rewards["foot_slip"].func = mdp.split_feet_slip
  # -0.3 -> -20 (2026-07-26). The contact-point-drift rewrite produces values
  # an order of magnitude smaller than the old site-velocity one (it no longer
  # counts a rolling foot as sliding), so the inherited weight left the term
  # realizing -0.002 -- entirely inert, for a failure mode that causes falls.
  # -20 restores it to ~-0.13 at current behaviour and scales with real slip.
  # -28 (was -20, 2026-07-27): realized only -0.33 of -17.1 budget, cheap
  # headroom -- same budget-ratio reasoning as flat_support/impact_vel above.
  cfg.rewards["foot_slip"].weight = -28.0
  # Smoothness pressure lives in joint (physical) space; action-space
  # rate/acc terms tax exploration noise itself and are kept small to avoid
  # premature std collapse. They use an L2-squared kernel on raw actions
  # (physical_rate = scale * raw_rate), so their weight must move with
  # scale^2 to preserve physical-space enforcement across a leg-scale
  # change. Rescaled by (7.0/1.5)^2 = 21.78 (2026-07-24, leg scale 1.5->7.0
  # to make ample hip/knee excursions reachable without walking the raw
  # action mean 15-35 std out -- see rhps1_constants.py's
  # _LEG_SCALE_MULTIPLIER comment): scale went up this time, so unlike the
  # previous 5.0->1.5 rescale these weights move up too, to keep the same
  # physical-space smoothness enforcement instead of quietly weakening it.
  # Split by phase below instead: stance/upper-body joints have different
  # smoothness needs than one blanket term captures.
  cfg.rewards.pop("action_acc_l2", None)
  
  # Smoothness consolidated into two terms (2026-07-26), replacing five that
  # all penalized "change" without distinguishing the two things we actually
  # care about:
  #   - action_jerk_l2: soft dynamics *while moving*. Second difference, in
  #     physical joint units, so an ample stride is cheap and a tremor is not.
  #     Upper body weighted 3x -- it has no reason to move much at all.
  #   - standing_joint_vel: near-total stillness *when stopped*, gated on the
  #     command so it never taxes the gait.
  # The five it replaces (action_rate_l2, stance_action_acc_l2,
  # upper_body_action_acc_l2, leg_joint_acc_l2 and the retained-but-separate
  # torque-rate term) together realized ~-14.7 per step against +10.5 for the
  # entire task, i.e. moving cost 1.4x more than succeeding paid. Penalizing
  # the first difference was the core error: it cannot ask for "soft but
  # moving", only for "less".
  cfg.rewards.pop("action_rate_l2", None)
  cfg.rewards.pop("stance_action_acc_l2", None)
  cfg.rewards.pop("upper_body_action_acc_l2", None)
  cfg.rewards.pop("leg_joint_acc_l2", None)
  # Weight and the upper-body coefficient are calibrated to reproduce, in
  # physical units, the strength the two acceleration terms this replaces had
  # in raw units: legs 150*1*0.049^2 = 0.360 against stance_action_acc_l2's
  # 0.3528, upper body 150*25*0.002^2 = 0.0150 against
  # upper_body_action_acc_l2's 0.0162. The 25x is not a preference invented
  # here -- it is what those two weights already implied once the action
  # scales are divided out, i.e. the old config was asking for a far stiffer
  # upper body than legs and simply expressing it through units.

  # action_jerk restored at -150 (2026-07-27). Briefly blamed for a collapse
  # and removed; that was wrong. Run 2026-07-26_20-47-53 ran this exact weight
  # and produced the best numbers of the session (flat contacts 2.91/4, sole
  # clearance 15.3 mm, fell_down 0.17). Removing it made things far worse
  # (fell_down 28.7 by iteration 3000), so it is load-bearing. The collapse it
  # was blamed for began with the standing changes below, not here -- the jerk
  # rising before fell_down was correlation, both being driven by the same
  # underlying instability.
  cfg.rewards["action_jerk"] = RewardTermCfg(
    func=mdp.action_jerk_l2,
    # -45 (was -150, 2026-07-27). Not a re-run of the earlier "blame the jerk"
    # mistake -- this is a whole-objective rescale, not a targeted cut. At iter
    # 1596 the realized budget was TOTAL+ 6.03 against TOTAL- 23.69, i.e. the
    # sum of penalties was ~4x the sum of task rewards, and action_jerk alone
    # (-9.77) outweighed every positive term combined. With a net-negative
    # per-step return, terminating early dominates surviving: the episode
    # sheds the whole remaining penalty stream and pays termination_penalty
    # once. fell_down climbing 0.73 -> 1.29 while *every* penalty magnitude
    # shrank and *every* task reward fell is exactly that signature. The five
    # dominant penalties are cut ~2.5x together (see flat_support,
    # joint_torques_l2, joint_torque_rate_l2, torque_limit_margin,
    # standing_joint_vel) to bring TOTAL- to roughly TOTAL+. Their ratios to
    # each other are unchanged, so this does not re-litigate any of the
    # relative-weight decisions above.
    # -90 (was -45, 2026-07-28): the robot still visibly vibrates. -45 came
    # from the whole-objective rescale, not from evidence that -45 is right;
    # -150 ran earlier this session with the best contact metrics of the
    # whole session and no collapse, so -90 is well inside proven-survivable
    # territory. Note this term is a *second* difference, so it targets
    # tremor specifically and is structurally blind to the slow sway --
    # that one is standing_base_motion's job, not a bigger weight here.
    # -45 (was -90, 2026-08-08). Retour a la valeur du reequilibrage global du
    # 2026-07-27, que le passage a -90 du lendemain avait defait sans re-mesurer
    # la balance. Mesure sur 2026-08-07_15-40-43 (iterations 6000-7200) : ce terme
    # realisait -28.33, soit 55.7 % du budget negatif, pour un total de -50.82
    # contre +8.95 de positif. Les termes qui faconnent la marche pesaient 1 a
    # 2 % et air_time realisait exactement zero.
    #
    # A -45 le total negatif tombe vers -36.6, donc encore 4x le positif : le
    # reequilibrage de juillet coupait CINQ termes ensemble. On n'en coupe qu'un
    # ici, faute de preuve sur les autres, et on mesure la balance au premier
    # jalon avant d'aller plus loin.
    weight=-45.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "coeffs": {r"CHEST|HEAD|SHOULDER|ELBOW|WRIST": 25.0},
    },
  )
  # New (2026-08-03): nothing with any leverage forced the stance back to nominal
  # at zero command, which is why the robot settles duck-footed -- pose's per-joint
  # exponential divided by 30 joints caps a 20 deg hip-yaw error at 0.05 reward.
  # Legs only: hip yaw and roll set where the feet point and how wide they sit,
  # ankle roll sets whether they end up flat. Hip/knee pitch are deliberately out,
  # so the term prescribes a stance without prescribing a squat depth.
  # -40 (was -4.0, 2026-08-04). The cost is 4 * error^2 where error is
  # Metrics/standing_pose_error, so the weight has to be set against the deviation
  # the robot actually reaches, not against the one it would reach if the term
  # already worked. Measured over 9 milestones of 2026-08-04_00-48-22, the error
  # climbs 0.043 -> 0.088 rad across 4500 iterations and decelerates: at -4.0 the
  # term realized -0.014 against standing_joint_vel's -0.47, i.e. 30x too quiet to
  # influence anything, and extrapolating its own trend it would have needed
  # ~17000 more iterations to reach the 0.28 rad where it starts to bite. At -40
  # it reaches 0.4 per step at 0.10 rad -- the level of the rest of the standing
  # family, without dominating it. That run is also the control proving this
  # config reproduces abl7 (9 milestones within 4%), so the comparison baseline
  # for judging this change is either run.
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
    # -0.7 (was -0.2, itself down from -0.5 in the 2026-07-27 uniform
    # rescale). Realized only -0.29 at -0.2, i.e. ~2.6% of the penalty
    # budget, while the robot visibly vibrates at standstill. Being an L2 sum
    # this only ever bites on *fast* residual motion -- the slow whole-body
    # sway is standing_base_motion's job below, not this one's.
    weight=-0.7,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # New (2026-07-28): nothing penalized base translation at zero command, and
  # both terms that could have (track_linear_velocity's wide 0.40 kernel,
  # standing_joint_vel's squared norm) go soft exactly in the slow-drift
  # regime -- see the function docstring for the arithmetic. L1 by design.
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
    cfg.events["foot_friction"].params["ranges"] = (0.5, 0.9)
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
    if "actor_history" in cfg.observations:
      cfg.observations["actor_history"].enable_corruption = False
    cfg.observations.pop("teacher", None)
    cfg.events.pop("push_robot", None)
    cfg.curriculum.pop("air_time_weight", None)
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
      params={"encoder_noise": 0.001},
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

  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0

  return cfg
