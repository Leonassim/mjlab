"""RHPS1 constants and helpers."""

import os
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import (
  ElmoChannelParams,
  ElmoReplicaActuatorCfg,
  ElmoReplicaDifferentialActuatorCfg,
  FiniteDifferencePdActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Native MJCF bundled with the repo. Points to on-disk meshes via
# `<compiler meshdir="../meshes">`. Override with MJLAB_RHPS1_XML if needed.
_BUNDLED_RHPS1_XML = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "RHPS1" / "xmls" / "RHPS1main.xml"
)
RHPS1_XML: Path = Path(os.environ.get("MJLAB_RHPS1_XML", str(_BUNDLED_RHPS1_XML)))


def _name_rhps1_main_collision_geoms(spec: mujoco.MjSpec) -> None:
  """Assign stable names to unnamed collision geoms from RHPS1main.xml.

  RHPS1main.xml uses `class="collision"` defaults but leaves many collision
  geoms unnamed. Our collision presets select geoms by regex on their names, so
  we synthesize names from the collision mesh names here instead of modifying
  the source XML.
  """

  existing_names = {geom.name for geom in spec.geoms if geom.name}
  for geom in spec.geoms:
    if geom.name:
      continue
    # Visual geoms already have collisions disabled in the MJCF defaults.
    if geom.contype == 0 and geom.conaffinity == 0:
      continue
    meshname = getattr(geom, "meshname", "")
    if not meshname:
      continue
    base_name = meshname[:-5] if meshname.endswith("_mesh") else meshname
    candidate = f"rhps1_collision_{base_name}"
    if candidate in existing_names:
      continue
    geom.name = candidate
    existing_names.add(candidate)


def _add_rhps1_foot_features(spec: mujoco.MjSpec) -> None:
  """Add RHPS1 foot sites/collisions expected by velocity tasks.

  `RHPS1main.xml` keeps only a simple unnamed sole box. For learning we restore
  the foot contact layout from `rhps1Hippolyte.xml`, but rename the split
  patches to the `left/right_foot1..4_collision` convention already used by the
  RHPS1 task config.
  """

  existing_geom_names = {geom.name for geom in spec.geoms if geom.name}

  def add_split_geoms(
    body: mujoco.MjsBody,
    side: str,
  ) -> None:
    if f"{side}_foot_collision" not in existing_geom_names:
      body.add_geom(
        name=f"{side}_foot_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{'L' if side == 'left' else 'R'}_ANKLE_P_LINK_mesh",
        pos=(0.0, 0.0, 0.0),
      )
    for name, pos, size in (
      # rhps1Hippolyte.xml uses URDF box sizes (full extents). MuJoCo box geoms
      # expect half-sizes, hence the division by two here.
      (
        f"{side}_foot1_collision",
        (0.08, 0.05 if side == "left" else 0.03, -0.09),
        (0.0525, 0.0275, 0.01),
      ),
      (
        f"{side}_foot2_collision",
        (0.08, -0.03 if side == "left" else -0.05, -0.09),
        (0.0525, 0.0275, 0.01),
      ),
      (
        f"{side}_foot3_collision",
        (-0.05, 0.05 if side == "left" else 0.03, -0.09),
        (0.0525, 0.0275, 0.01),
      ),
      (
        f"{side}_foot4_collision",
        (-0.05, -0.03 if side == "left" else -0.05, -0.09),
        (0.0525, 0.0275, 0.01),
      ),
    ):
      if name not in existing_geom_names:
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)

  left_ankle = spec.body("L_ANKLE_P_LINK")
  right_ankle = spec.body("R_ANKLE_P_LINK")
  site_size = (0.001, 0.001, 0.001)
  existing_site_names = {site.name for site in spec.sites}

  def add_site_if_missing(
    body: mujoco.MjsBody,
    name: str,
    pos: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
  ) -> None:
    if name in existing_site_names:
      return
    body.add_site(name=name, pos=pos, size=site_size, rgba=rgba)
    existing_site_names.add(name)

  # Sites used by RHPS1 rewards/sensors.
  for body, side_sign, prefix in (
    (left_ankle, 1.0, "left"),
    (right_ankle, -1.0, "right"),
  ):
    # z = -0.10 puts this at the actual sole plane: the split contact boxes
    # sit at z = -0.09 with a 0.01 half-height, so their underside -- the
    # surface the robot really walks on, carrying 100% of the load (the
    # ankle-link mesh geom measures 0 N) -- is exactly -0.10. It used to be
    # -0.08, i.e. 20 mm up the ankle, which silently offset every consumer:
    # Metrics/peak_height_mean read 0.022 for a foot whose sole was 2 mm off
    # the ground, and min_foot_height's 0.030 "target" was really asking for
    # 10 mm of clearance. Measuring at the sole also makes foot_slip and
    # gait_phase's stance term read the velocity of the contacting surface
    # rather than of a point 20 mm above it.
    add_site_if_missing(body, f"{prefix}_foot", (0.0, 0.0, -0.10), (1, 0, 0, 1))
    add_site_if_missing(
      body, f"{prefix}_foot_toes", (0.08, 0.0, -0.08), (0.5, 0.5, 0.5, 0.3)
    )
    add_site_if_missing(
      body, f"{prefix}_foot_heel", (-0.08, 0.0, -0.08), (0.5, 0.5, 0.5, 0.3)
    )
    # Inner/outer were not explicit named sites in rhps1Hippolyte.xml, so we
    # keep them as lightweight helpers for the existing RHPS1 rewards.
    add_site_if_missing(
      body,
      f"{prefix}_foot_inner",
      (0.015, 0.04 * side_sign, -0.08),
      (0.5, 0.5, 0.5, 0.3),
    )
    add_site_if_missing(
      body,
      f"{prefix}_foot_outer",
      (0.015, -0.04 * side_sign, -0.08),
      (0.5, 0.5, 0.5, 0.3),
    )

  # Extra split foot contact sites from rhps1Hippolyte.xml.
  add_site_if_missing(
    left_ankle, "left_foot_left_toes", (0.12, -0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    left_ankle, "left_foot_left_heel", (-0.09, -0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    left_ankle, "left_foot_right_toes", (0.12, 0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    left_ankle, "left_foot_right_heel", (-0.09, 0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    right_ankle, "right_foot_left_toes", (0.12, -0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    right_ankle, "right_foot_left_heel", (-0.09, -0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    right_ankle, "right_foot_right_toes", (0.12, 0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )
  add_site_if_missing(
    right_ankle, "right_foot_right_heel", (-0.09, 0.06, -0.05), (0.5, 0.5, 0.5, 0.3)
  )

  # Named contact geoms expected by RHPS1 collision presets, using the
  # left/right toe/heel patch layout from rhps1Hippolyte.xml.
  add_split_geoms(left_ankle, "left")
  add_split_geoms(right_ankle, "right")


def get_spec() -> mujoco.MjSpec:
  """Load the RHPS1 MJCF and add task-specific features."""
  if not RHPS1_XML.exists():
    raise FileNotFoundError(f"RHPS1 MJCF is missing at {RHPS1_XML}.")

  spec = mujoco.MjSpec.from_file(str(RHPS1_XML))
  if RHPS1_XML.name == "RHPS1main.xml":
    _name_rhps1_main_collision_geoms(spec)
  # Fix invalid inertias from URDF by balancing the inertia tensor.
  spec.compiler.balanceinertia = True
  # The deployment QP monitors shoulder-chest self-collision pairs, but the
  # bundled XML excludes them, which also hides them from the proximity
  # sensors. Re-enable so the leg/arm proximity penalties see the same pairs
  # as the QP (hulls sit ~2.2 cm apart at rest; forces need penetration).
  _qp_monitored = {
    frozenset(("L_SHOULDER_Y_LINK", "CHEST_P_LINK")),
    frozenset(("R_SHOULDER_Y_LINK", "CHEST_P_LINK")),
  }
  for exclude in [e for e in spec.excludes]:
    if frozenset((exclude.bodyname1, exclude.bodyname2)) in _qp_monitored:
      spec.delete(exclude)
  _add_rhps1_foot_features(spec)

  existing_sensor_names = {sensor.name for sensor in spec.sensors}

  def add_sensor_if_missing(**kwargs) -> None:
    if kwargs["name"] in existing_sensor_names:
      return
    spec.add_sensor(**kwargs)
    existing_sensor_names.add(kwargs["name"])

  # Add velocimeters for impact velocity reward.
  for name, site in (
    ("left_foot_lin_vel", "left_foot"),
    ("left_foot_toes_lin_vel", "left_foot_toes"),
    ("left_foot_heel_lin_vel", "left_foot_heel"),
    ("left_foot_inner_lin_vel", "left_foot_inner"),
    ("left_foot_outer_lin_vel", "left_foot_outer"),
    ("right_foot_lin_vel", "right_foot"),
    ("right_foot_toes_lin_vel", "right_foot_toes"),
    ("right_foot_heel_lin_vel", "right_foot_heel"),
    ("right_foot_inner_lin_vel", "right_foot_inner"),
    ("right_foot_outer_lin_vel", "right_foot_outer"),
    ("left_foot_left_toes_lin_vel", "left_foot_left_toes"),
    ("left_foot_left_heel_lin_vel", "left_foot_left_heel"),
    ("left_foot_right_toes_lin_vel", "left_foot_right_toes"),
    ("left_foot_right_heel_lin_vel", "left_foot_right_heel"),
    ("right_foot_left_toes_lin_vel", "right_foot_left_toes"),
    ("right_foot_left_heel_lin_vel", "right_foot_left_heel"),
    ("right_foot_right_toes_lin_vel", "right_foot_right_toes"),
    ("right_foot_right_heel_lin_vel", "right_foot_right_heel"),
  ):
    add_sensor_if_missing(
      name=name,
      type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname=site,
    )
  add_sensor_if_missing(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="BODY",
  )

  # Match H1 grouping convention:
  # - visual geoms in group 2
  # - collision geoms in group 3
  # - sites in group 4
  #
  # RHPS1main.xml already labels geoms semantically through the MJCF defaults
  # `class="visual"` and `class="collision"`. Preserve that split instead of
  # inferring from geom names, otherwise unnamed convex collision meshes end up
  # mixed with the pretty visuals in group 2.
  for geom in spec.geoms:
    if geom.conaffinity == 0 and geom.contype == 0:
      geom.group = 2
    else:
      geom.group = 3
  for site in spec.sites:
    site.group = 4

  # Disable all collisions by default. Collision presets below will re-enable
  # the selected geom sets.
  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0
  return spec


##
# Actuator config.
##

# armature=1.0 everywhere is DELIBERATE, not a missing value. The 20000/400 PD
# gains emulate a stiff position servo and were tuned with armature=1.0; they are
# not separable from it. With the real armature the explicit-integration criterion
# kd*dt/M reaches 13 against a threshold of 2, and the joint either diverges or
# buzzes pinned to its effort limit. A smaller timestep does not save it (worst
# case needs dt <= 76 us) and implicitfast cannot help, since MuJoCo will not
# implicitly integrate a torque supplied through ctrl. Enabling the real values
# requires retuning kp and kd per joint, hence redoing action_scale. Each joint
# records its real value next to its armature line.
RHPS1_ACTUATOR_CROTCH_Y = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_CROTCH_Y",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=35.0,
  # Real armature: 0.07087
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=8.0,
  velocity_limits=8.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_CROTCH_P = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_CROTCH_P",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=140.0,
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=8.0,
  velocity_limits=8.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_CROTCH_R = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_CROTCH_R",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=100.0,
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=8.0,
  velocity_limits=8.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

# effort_limit below: flat 70.0 on both sides (was the real per-side ELMO
# current-limit-derived bound, 110.9 L / 69.3 R -- see git history). Reverted:
# that derivation (tau_max = eta*N*Kt*i_limit) is only valid for the real
# robot's actual cascaded P(pos)/PI(vel)+current-saturation loop
# (ElmoReplicaActuator), not for the flat-gain PD + MuJoCo effort clamp this
# config actually trains with -- the two aren't the same dynamical system, so
# a limit derived for one doesn't have a principled meaning applied to the
# other. Until the real actuator replica is what's actually training (see
# elmo_replica_actuator.py), 70.0 (the more conservative of the two real
# per-side numbers) on both knees is the honest placeholder.
RHPS1_ACTUATOR_KNEE_L = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r"L_KNEE_P",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=70.0,
  # Real armature: 0.27651
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=10.0,
  velocity_limits=10.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_KNEE_R = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r"R_KNEE_P",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=70.0,
  # Real armature: 0.27651
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=10.0,
  velocity_limits=10.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_ANKLE_P = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_ANKLE_P",),
  stiffness=10000.0,
  damping=300.0,
  effort_limit=65.0,
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=10.0,
  velocity_limits=10.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_ANKLE_R = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_ANKLE_R",),
  stiffness=10000.0,
  damping=300.0,
  effort_limit=45.0,
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=10.0,
  velocity_limits=10.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_TORSO = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r"CHEST_.*",),
  stiffness=44000.0,
  damping=440.0,
  effort_limit=120.0,
  # Real armature: 0.03994
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_SHOULDER_Y = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_SHOULDER_Y",),
  stiffness=14000.0,
  damping=240.0,
  effort_limit=50.0,
  # Real armature: 0.02556
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_SHOULDER_P = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_SHOULDER_P",),
  stiffness=15000.0,
  damping=240.0,
  effort_limit=50.0,
  # Real armature: 0.0312
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_SHOULDER_R = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_SHOULDER_R",),
  stiffness=14000.0,
  damping=240.0,
  effort_limit=50.0,
  # Real armature: 0.02556
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_ELBOW_P = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_ELBOW_P",),
  stiffness=14000.0,
  damping=240.0,
  effort_limit=40.0,
  # Real armature: 0.0264
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_ELBOW_Y = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_ELBOW_Y",),
  stiffness=14000.0,
  damping=240.0,
  effort_limit=40.0,
  # Real armature: 0.0264
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_WRIST = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_WRIST_.*",),
  stiffness=14000.0,
  damping=240.0,
  effort_limit=30.0,
  # Real armature: 0.01485
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=6.0,
  velocity_limits=6.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_HAND = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_HAND",),
  stiffness=500.0,
  damping=5.0,
  effort_limit=15.0,
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=3.0,
  velocity_limits=3.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATOR_HEAD = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r"HEAD_.*",),
  stiffness=2000.0,
  damping=50.0,
  effort_limit=13.0,
  # Real armature: 0.00331
  armature=1.0,
  position_target_filter_alpha=0.0,
  velocity_target_limit=4.0,
  velocity_limits=4.0,
  velocity_damper_di=0.4,
  velocity_damper_ds=0.01,
  velocity_damper_vel_percent=0.9,
)

RHPS1_ACTUATORS: tuple[FiniteDifferencePdActuatorCfg, ...] = (
  RHPS1_ACTUATOR_CROTCH_Y,
  RHPS1_ACTUATOR_CROTCH_P,
  RHPS1_ACTUATOR_CROTCH_R,
  RHPS1_ACTUATOR_KNEE_L,
  RHPS1_ACTUATOR_KNEE_R,
  RHPS1_ACTUATOR_ANKLE_P,
  RHPS1_ACTUATOR_ANKLE_R,
  RHPS1_ACTUATOR_TORSO,
  RHPS1_ACTUATOR_SHOULDER_Y,
  RHPS1_ACTUATOR_SHOULDER_P,
  RHPS1_ACTUATOR_SHOULDER_R,
  RHPS1_ACTUATOR_ELBOW_P,
  RHPS1_ACTUATOR_ELBOW_Y,
  RHPS1_ACTUATOR_WRIST,
  RHPS1_ACTUATOR_HEAD,
)

##
# Real ELMO-drive-replica actuators (P(pos)/PI(vel) cascade + current saturation +
# anti-windup, see mjlab.actuator.elmo_replica_actuator / elmo_replica_differential_actuator).
#
# NOT ACTIVATED: these cfgs exist so the wiring/values are ready, but are not part of
# RHPS1_ACTUATORS above -- the FiniteDifferencePdActuatorCfg-based entries still drive
# training. Swap them in (and re-tune/re-train) deliberately, not as a drive-by change.
#
# Covers the 22 ACTUATOR_TYPE_ROTATE joints only (both knees, crotch_Y, chest, head,
# shoulders, elbows, wrists). The 8 linear-actuator joints (hip roll/pitch, ankle
# roll/pitch) are NOT covered -- real per-drive current limits are known for them too
# (RHPS1_gains CSV), but converting drive-side force to joint-side torque needs the
# parallel-cylinder attachment-point geometry, which isn't in any file we have access to
# (see RHPS1_gains/README.md "Actuator Limits (Real Values, All Joints)").
#
# All numeric values below are real, from RHPS1_gains/FromRealRobot/drive_gains_map.csv
# (Kp_pos<-KP3, Kp_vel<-KP2, Ki_vel<-KI2, gear_ratio<-gear_ratio_N, torque_constant<-
# torque_constant_Nm_per_Arms, current_limit_continuous<-current_limit_continuous_A,
# current_limit_peak<-current_limit_peak_A), not placeholders. eta is left at the
# ElmoReplicaActuatorCfg/ElmoChannelParams default (1.0, an upper bound) throughout --
# the ~0.77 calibrated estimate is knee-specific (see RHPS1_gains's notebook "eta
# calibration" section) and hasn't been cross-checked for the other 20 joints, so it is
# NOT applied here; revisit before activating.
##

RHPS1_ELMO_ACTUATOR_L_CROTCH_Y = ElmoReplicaActuatorCfg(
  target_names_expr=("L_CROTCH_Y",),
  Kp_pos=30.0, Kp_vel=1.1e-6, Ki_vel=45.0,
  gear_ratio=159.0907, torque_constant=0.0582,
  current_limit_continuous=0.88, current_limit_peak=2.55,
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_R_CROTCH_Y = ElmoReplicaActuatorCfg(
  target_names_expr=("R_CROTCH_Y",),
  Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0,
  gear_ratio=159.0907, torque_constant=0.0582,
  current_limit_continuous=2.16, current_limit_peak=4.24,
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_L_KNEE_P = ElmoReplicaActuatorCfg(
  target_names_expr=("L_KNEE_P",),
  Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0,
  gear_ratio=210.0, torque_constant=0.101,
  current_limit_continuous=2.94, current_limit_peak=6.79,
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_R_KNEE_P = ElmoReplicaActuatorCfg(
  target_names_expr=("R_KNEE_P",),
  Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0,
  gear_ratio=210.0, torque_constant=0.101,
  current_limit_continuous=2.16, current_limit_peak=4.24,
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_L_SHOULDER_P = ElmoReplicaActuatorCfg(
  target_names_expr=("L_SHOULDER_P",),
  Kp_pos=39.0, Kp_vel=1.1e-6, Ki_vel=45.0,
  gear_ratio=200.0, torque_constant=0.0470,
  current_limit_continuous=0.88, current_limit_peak=1.68,
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_R_SHOULDER_P = ElmoReplicaActuatorCfg(
  target_names_expr=("R_SHOULDER_P",),
  Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0,
  gear_ratio=199.9998, torque_constant=0.0470,
  current_limit_continuous=2.94, current_limit_peak=6.79,
  armature=1.0,
)

RHPS1_ELMO_ACTUATOR_CHEST = ElmoReplicaDifferentialActuatorCfg(
  # dof order (target_names_expr) = (CHEST_P, CHEST_Y); ch_a=ChestYPL=P+Y, ch_b=ChestYPR=P-Y
  target_names_expr=("CHEST_P", "CHEST_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=160.0, torque_constant=0.0470,
    current_limit_continuous=2.94, current_limit_peak=6.79,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=160.0, torque_constant=0.0470,
    current_limit_continuous=2.94, current_limit_peak=6.79,
  ),
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_HEAD = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (HEAD_P, HEAD_Y); ch_a=HeadYPL=P+Y, ch_b=HeadYPR=P-Y
  target_names_expr=("HEAD_P", "HEAD_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=70.8332, torque_constant=0.0458,
    current_limit_continuous=2.16, current_limit_peak=4.24,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=70.8332, torque_constant=0.0458,
    current_limit_continuous=2.16, current_limit_peak=4.24,
  ),
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_L_SHOULDER_RY = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (L_SHOULDER_R, L_SHOULDER_Y); ch_a=LShoulderRYF=R+Y, ch_b=LShoulderRYB=R-Y
  target_names_expr=("L_SHOULDER_R", "L_SHOULDER_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=39.0, Kp_vel=1.1e-6, Ki_vel=45.0, gear_ratio=166.6667, torque_constant=0.0487,
    current_limit_continuous=0.88, current_limit_peak=1.68,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=39.0, Kp_vel=1.1e-6, Ki_vel=45.0, gear_ratio=166.6667, torque_constant=0.0487,
    current_limit_continuous=0.88, current_limit_peak=2.56,
  ),
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_R_SHOULDER_RY = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (R_SHOULDER_R, R_SHOULDER_Y); ch_a=RShoulderRYF=R+Y, ch_b=RShoulderRYB=R-Y
  target_names_expr=("R_SHOULDER_R", "R_SHOULDER_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=166.6667, torque_constant=0.0487,
    current_limit_continuous=2.16, current_limit_peak=2.83,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=166.6667, torque_constant=0.0487,
    current_limit_continuous=2.16, current_limit_peak=2.83,
  ),
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_L_ELBOW_PY = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (L_ELBOW_P, L_ELBOW_Y); ch_a=LElbowPYO=P+Y, ch_b=LElbowPYI=P-Y
  target_names_expr=("L_ELBOW_P", "L_ELBOW_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=199.9998, torque_constant=0.0458,
    current_limit_continuous=2.94, current_limit_peak=6.79,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=39.0, Kp_vel=1.1e-6, Ki_vel=45.0, gear_ratio=200.0001, torque_constant=0.0458,
    current_limit_continuous=0.88, current_limit_peak=2.56,
  ),
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_R_ELBOW_PY = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (R_ELBOW_P, R_ELBOW_Y); ch_a=RElbowPYI=P+Y, ch_b=RElbowPYO=P-Y
  # NOTE: mirrored vs L_ELBOW (channel_a/b swapped which physical drive plays which
  # role) -- verified against TransListenerEx.hpp, not a typo.
  target_names_expr=("R_ELBOW_P", "R_ELBOW_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=39.0, Kp_vel=1.1e-6, Ki_vel=45.0, gear_ratio=200.0001, torque_constant=0.0458,
    current_limit_continuous=0.71, current_limit_peak=2.05,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=50.0, Kp_vel=4.3e-6, Ki_vel=0.02, gear_ratio=199.9998, torque_constant=0.0458,
    current_limit_continuous=1.03, current_limit_peak=2.03,
  ),
  armature=1.0,
)
# L/R_WRIST: the real robot's own joint names here are L/R_WRIST_R (Roll) + L/R_WRIST_Y
# (Yaw) -- matching RHPS1main.urdf and RHPS1_REF_JOINT_ORDER below -- NOT the "P" the
# motor channel names ("WristPYI/PYO") and RHPS1_gains's raw CSV joint_a column suggest.
# That CSV labeling was deliberately NOT "fixed": it may be correct for a different
# hand/end-effector configuration than the one this file targets (RHPS1_gains's URDF
# check confirmed no _WRIST_P joint exists in ITS bundled URDF, but wrist DOF naming can
# vary by robot config/hand). The dof_a/dof_b assignment below uses this repo's own
# RHPS1_REF_JOINT_ORDER naming (_WRIST_R), independent of that CSV column.
RHPS1_ELMO_ACTUATOR_L_WRIST_RY = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (L_WRIST_R, L_WRIST_Y); ch_a=LWristPYI=R+Y, ch_b=LWristPYO=R-Y
  target_names_expr=("L_WRIST_R", "L_WRIST_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=150.0003, torque_constant=0.0458,
    current_limit_continuous=2.94, current_limit_peak=6.79,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=60.0, Kp_vel=5.0e-6, Ki_vel=2.0, gear_ratio=150.0, torque_constant=0.0458,
    current_limit_continuous=2.94, current_limit_peak=6.79,
  ),
  armature=1.0,
)
RHPS1_ELMO_ACTUATOR_R_WRIST_RY = ElmoReplicaDifferentialActuatorCfg(
  # dof order = (R_WRIST_R, R_WRIST_Y); ch_a=RWristPYI=R+Y, ch_b=RWristPYO=R-Y
  target_names_expr=("R_WRIST_R", "R_WRIST_Y"),
  channel_a=ElmoChannelParams(
    Kp_pos=39.0, Kp_vel=1.1e-6, Ki_vel=45.0, gear_ratio=150.0, torque_constant=0.0458,
    current_limit_continuous=0.71, current_limit_peak=2.05,
  ),
  channel_b=ElmoChannelParams(
    Kp_pos=50.0, Kp_vel=4.3e-6, Ki_vel=0.02, gear_ratio=150.0003, torque_constant=0.0458,
    current_limit_continuous=1.03, current_limit_peak=2.03,
  ),
  armature=1.0,
)

RHPS1_ELMO_ACTUATORS_INACTIVE = (
  RHPS1_ELMO_ACTUATOR_L_CROTCH_Y,
  RHPS1_ELMO_ACTUATOR_R_CROTCH_Y,
  RHPS1_ELMO_ACTUATOR_L_KNEE_P,
  RHPS1_ELMO_ACTUATOR_R_KNEE_P,
  RHPS1_ELMO_ACTUATOR_L_SHOULDER_P,
  RHPS1_ELMO_ACTUATOR_R_SHOULDER_P,
  RHPS1_ELMO_ACTUATOR_CHEST,
  RHPS1_ELMO_ACTUATOR_HEAD,
  RHPS1_ELMO_ACTUATOR_L_SHOULDER_RY,
  RHPS1_ELMO_ACTUATOR_R_SHOULDER_RY,
  RHPS1_ELMO_ACTUATOR_L_ELBOW_PY,
  RHPS1_ELMO_ACTUATOR_R_ELBOW_PY,
  RHPS1_ELMO_ACTUATOR_L_WRIST_RY,
  RHPS1_ELMO_ACTUATOR_R_WRIST_RY,
)  # 6 solo + 8*2 differential = 22 joints. NOT referenced by RHPS1_ACTUATORS.

##
# Reference joint order from mc_rhps1 (useful when wiring observations/actions).
RHPS1_REF_JOINT_ORDER = [
  "L_CROTCH_Y",
  "L_CROTCH_R",
  "L_CROTCH_P",
  "L_KNEE_P",
  "L_ANKLE_R",
  "L_ANKLE_P",
  "CHEST_Y",
  "CHEST_P",
  "R_CROTCH_Y",
  "R_CROTCH_R",
  "R_CROTCH_P",
  "R_KNEE_P",
  "R_ANKLE_R",
  "R_ANKLE_P",
  "HEAD_Y",
  "HEAD_P",
  "L_SHOULDER_P",
  "L_SHOULDER_R",
  "L_SHOULDER_Y",
  "L_ELBOW_P",
  "L_ELBOW_Y",
  "L_WRIST_R",
  "L_WRIST_Y",
  "R_SHOULDER_P",
  "R_SHOULDER_R",
  "R_SHOULDER_Y",
  "R_ELBOW_P",
  "R_ELBOW_Y",
  "R_WRIST_R",
  "R_WRIST_Y",
]

# Keyframe / initial state derived from mc_rhps1 _stance (deg -> rad), legs
# re-solved via mujoco FK for a straighter knee to keep standing torque low
# on the real robot's fragile knees: foot sole flat (pitch chain sums to
# zero) and hip-to-ankle offset unchanged, so this is a pure redistribution
# of flexion, not a change in standing height or forward lean.
RHPS1_INIT_STATE = EntityCfg.InitialStateCfg(
  # Base height goes with the keyframe: this is policy 0's (knee 0.622), more
  # flexed than the straight-knee one, so the pelvis sits 1.3 cm lower. Move both
  # together or the robot starts interpenetrating or in free fall.
  pos=(0.0, 0.0, 0.837656),
  joint_pos={
    "R_CROTCH_Y": 0.010533,
    "R_CROTCH_R": -0.028787,
    "R_CROTCH_P": -0.271191,
    "R_KNEE_P": 0.622021,
    "R_ANKLE_R": 0.030652,
    "R_ANKLE_P": -0.350679,
    "L_CROTCH_Y": -0.010533,
    "L_CROTCH_R": 0.028787,
    "L_CROTCH_P": -0.271191,
    "L_KNEE_P": 0.622021,
    "L_ANKLE_R": -0.030652,
    "L_ANKLE_P": -0.350679,
    "CHEST_Y": 0.0,
    "CHEST_P": 0.0,
    "HEAD_Y": 0.0,
    "HEAD_P": 0.0,
    "R_SHOULDER_P": 0.261799,
    "R_SHOULDER_R": -0.174533,
    "R_SHOULDER_Y": 0.087266,
    "R_ELBOW_P": -0.523599,
    "R_ELBOW_Y": 0.0,
    "R_WRIST_R": 0.0,
    "R_WRIST_Y": 0.0,
    "L_SHOULDER_P": 0.261799,
    "L_SHOULDER_R": 0.174533,
    "L_SHOULDER_Y": -0.087266,
    "L_ELBOW_P": -0.523599,
    "L_ELBOW_Y": 0.0,
    "L_WRIST_R": 0.0,
    "L_WRIST_Y": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_RHPS1_FOOT_COLLISION_EXPR = r"^(left|right)_foot([1-4])_collision$"
_RHPS1_BODY_COLLISION_EXPR = r"^rhps1_collision_.*$"
_RHPS1_ALL_COLLISION_EXPR = r"^((left|right)_foot([1-4])_collision|rhps1_collision_.*)$"

# Feet-ground contacts only.
RHPS1_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_RHPS1_FOOT_COLLISION_EXPR,),
  condim=3,
  priority=1,
  friction=(0.5,),
  disable_other_geoms=False,
)

# Links carrying the deployment QP's minimalSelfCollisions pairs. Legs use
# ANKLE_R as a proxy for the ANKLE_P (foot) hulls: foot geoms cannot take a
# collision gap without corrupting ground-contact/air-time sensing.
_RHPS1_LEG_LINK_COLLISION_EXPR = r"^rhps1_collision_[LR]_(CROTCH_P|KNEE_P|ANKLE_R)_LINK$"
_RHPS1_QP_PAIR_COLLISION_EXPR = (
  r"^rhps1_collision_([LR]_(CROTCH_P|KNEE_P|ANKLE_R|SHOULDER_Y|ELBOW_Y|WRIST_Y)_LINK"
  r"|CHEST_P_LINK|BODY)$"
)

# mujoco-warp semantics: contacts produce force when dist < margin and are
# *detected* (visible to contact sensors, forceless) when dist < margin + gap.
# A pure gap therefore exposes leg-leg clearance to the proximity penalty
# without altering the physics. Pair detection range ~= sum of both geoms'
# gaps (~5 cm), comfortably beyond the 2 cm penalty threshold that keeps the
# gait outside the deployment QP's self-collision damper zone.
_RHPS1_LEG_PROXIMITY_GAP = 0.025

# Enable all named collision geoms, including self-collisions.
RHPS1_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_RHPS1_ALL_COLLISION_EXPR,),
  condim={_RHPS1_FOOT_COLLISION_EXPR: 3, r"^rhps1_collision_.*$": 1},
  priority={_RHPS1_FOOT_COLLISION_EXPR: 1},
  friction={_RHPS1_FOOT_COLLISION_EXPR: (0.5,)},
  # Feet keep the default world-collision bit (1). Body geoms use a separate
  # bit (2) so they can self-collide without taking over terrain contacts.
  contype={_RHPS1_FOOT_COLLISION_EXPR: 1, _RHPS1_BODY_COLLISION_EXPR: 2},
  conaffinity={_RHPS1_FOOT_COLLISION_EXPR: 1, _RHPS1_BODY_COLLISION_EXPR: 2},
  gap={_RHPS1_QP_PAIR_COLLISION_EXPR: _RHPS1_LEG_PROXIMITY_GAP},
  disable_other_geoms=False,
)

# Enable world/body collisions while avoiding robot self-collisions.
RHPS1_FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(_RHPS1_ALL_COLLISION_EXPR,),
  contype=0,
  conaffinity=1,
  condim={_RHPS1_FOOT_COLLISION_EXPR: 3, r"^rhps1_collision_.*$": 1},
  priority={_RHPS1_FOOT_COLLISION_EXPR: 1},
  friction={_RHPS1_FOOT_COLLISION_EXPR: (0.5,)},
  disable_other_geoms=False,
)

# Default collision mode.
RHPS1_COLLISION = RHPS1_FULL_COLLISION

##
# Final config.
##

RHPS1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=RHPS1_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_rhps1_robot_cfg() -> EntityCfg:
  """Return a fresh RHPS1 EntityCfg. Fill TODOs before using in training."""
  return EntityCfg(
    init_state=RHPS1_INIT_STATE,
    collisions=(RHPS1_COLLISION,),
    spec_fn=get_spec,
    articulation=RHPS1_ARTICULATION,
  )


# Project the position target so the PD demand stays inside
# ratio * effort_limit, instead of letting MuJoCo clamp it silently.
#
# At ratio 1.0 this is not a new constraint: tau is affine and increasing in
# q*, so clamping tau and projecting q* onto its preimage are the same
# operation -- same torque, same dynamics. What it buys is deployment, where
# mc_rtc has no torque clamp downstream of the PD: a target that only works
# because something clips it does not transfer.
#
# The half-window is e/kp = 0.143 action units on every joint, so one action
# unit is 7x the window and the projection bites on nearly every step by
# construction. Do not judge it before iteration 1000.
#
# kp cannot be lowered to widen the window: it is the robot's low-level
# position gain, not a simulation knob.

# mc_rtc QP PostureTask stiffness, reproduced upstream of the PD.
#
# 1600 is the same value on both sides of the transfer: on the robot and in
# mc_mujoco. Both give sqrt(K)*dt = 0.20, the number governing discrete-time
# stability -- 1600 at 200 Hz, 40000 at 1 kHz. Training previously sent the
# policy output straight to the PD, skipping the ~25 ms this second order adds
# on the robot.
_POSTURE_TASK_STIFFNESS = 1600.0

_TORQUE_FEASIBILITY_RATIO = 1.0

# Control period: one policy step. Must equal the env's step_dt
# (sim.mujoco.timestep * decimation = 0.0025 * 2). env_cfgs.py asserts this
# rather than trusting the comment -- the action scale below is derived from it,
# so a silent mismatch would mis-scale every joint.
_CONTROL_DT = 0.005
# EMA on the finite-difference velocity target.
#
# It is hidden state: qd* under an EMA depends on the whole history of targets
# and nothing in the observation carries it, so the action -> torque map is not
# a function of what the policy sees. Kept at 0.8 anyway because 0.0 was
# ablated and rejected: differentiating per-step exploration noise multiplies
# it by 1/dt = 200, and demand rose (pd_demand_ratio 2.93 -> 4.15, falls 7-13x
# worse). The open problem is recovering the motion 0.0 bought without that
# torque bill.
#
# The C++ controller must match (mc_rtc key vel_target_filter_alpha) or the
# deployed qd* is a different signal.
_VELOCITY_TARGET_FILTER_ALPHA = 0.8

for a in RHPS1_ARTICULATION.actuators:
  assert isinstance(a, FiniteDifferencePdActuatorCfg)
  a.torque_feasibility_ratio = _TORQUE_FEASIBILITY_RATIO
  a.velocity_target_filter_alpha = _VELOCITY_TARGET_FILTER_ALPHA
  a.posture_task_stiffness = _POSTURE_TASK_STIFFNESS

# One raw action unit = the instantaneous change in position target that, on its
# own, produces exactly the effort limit:
#
#     tau = kp*(q* - q) + kd*(qd* - qd),  qd* = (q*_k - q*_{k-1}) / dt
#
# so a one-step jump of dq in the target, from a settled joint, demands
# (kp + kd/dt)*dq. Setting that to the effort limit gives
#
#     scale = effort_limit / (kp + kd/dt)
#
# This replaces effort_limit/stiffness, which only accounted for the kp term and
# therefore ignored the *dominant* one: kd/dt is 80000 against kp=20000 at the
# hip and 60000 against 10000 at the ankle, i.e. 4x to 6x larger. The old
# formula overstated the usable range by that factor, and unevenly across joints
# -- the ankles were getting ~30% more authority relative to the hips than their
# actuators justify, purely as an artifact of which term was counted.
#
# The statistically exact version for iid exploration noise (the target is
# first-differenced, so consecutive noise samples anticorrelate) is
# e / sqrt((kp + kd/dt)^2 + (kd/dt)^2), uniformly 1.28-1.32x tighter than this.
# Not used: it says the same thing about relative per-joint weighting, the
# absolute factor is absorbed by init_std anyway, and a bound on a single action
# ("one unit saturates") is easier to reason about and to reproduce in the
# controller than a statement about a noise distribution.
#
# Depends on dt, so it depends on the EMA being off -- with a filter the
# instantaneous kd response is (1-alpha)*kd/dt instead, and this formula would
# understate the range by ~2x. See velocity_target_filter_alpha above.
RHPS1_ACTION_SCALE: dict[str, float] = {}
for a in RHPS1_ARTICULATION.actuators:
  assert isinstance(a, FiniteDifferencePdActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  d = a.damping
  names = a.target_names_expr
  assert e is not None
  for n in names:
    RHPS1_ACTION_SCALE[n] = e / s

# Upper-body joints: capped below saturation (effort/stiffness ≈ 0.003 for shoulders).
# 0.002 keeps torques well under effort limits and reduces oscillation.
upper_scale = 0.002
upper_keys = [
  k
  for k in RHPS1_ACTION_SCALE
  if any(tok in k for tok in ("CHEST", "SHOULDER", "ELBOW", "WRIST", "HAND", "HEAD"))
]
for k in upper_keys:
  RHPS1_ACTION_SCALE.pop(k, None)

for name in (
  "CHEST_Y", "CHEST_P", "HEAD_Y", "HEAD_P",
  "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P", "L_ELBOW_Y",
  "L_WRIST_R", "L_WRIST_Y",
  "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", "R_ELBOW_Y",
  "R_WRIST_R", "R_WRIST_Y",
):
  RHPS1_ACTION_SCALE[name] = upper_scale

# With the policy's unit-std Gaussian exploration, this scale IS the
# exploration amplitude in joint space -- never drive it with a curriculum,
# unlike reward-shaping terms, since it redefines what a raw network output
# means mid-training. Deployment note: the controller yaml action_scale
# must match the training that produced the deployed ONNX.
#
# 1.5 -> 7.0 (2026-07-24): CROTCH_P's scale at 1.5 was only ~0.0105 rad per
# raw-action unit, so a hip-flexion excursion in the 0.3-0.4 rad range (the
# ample, high-knee stride the reward shaping was pushing for) needed raw
# actions around 30-70 -- 15-35 std out from a std=2.0 policy, essentially
# unreachable through normal gradient progress, since torque/smoothness/pose
# penalties all pull the raw-action mean back toward 0 the whole way out.
# 7.0 puts CROTCH_P at ~0.049 rad/unit, so that same 0.3-0.4 rad excursion
# needs a raw action around 6-8: reachable without a multi-hundred-iteration
# walk through penalty-dominated territory. See rewards.py's
# action_rate_l2/stance_action_acc_l2 comment -- their weights must move
# with scale^2 (up, not down, since scale increased) to keep the same
# physical-space smoothness enforcement.
#
# 7.0 -> 1.0 (2026-07-29), paired with the torque_feasibility_ratio set on
# every leg actuator below -- neither change is useful without the other.
#
# What 7.0 cost: the PD demand is clamped silently at the effort limit, so
# every action asking for more than the limit gives the same torque, the same
# dynamics and the same reward. PPO is model-free, nothing differentiates
# through that clamp, so the whole saturated region is one flat plateau in the
# return. Measured consequence: Metrics/pd_demand_ratio_mean 2.7-5.6 (max 350),
# i.e. the policy routinely asked for several times the deliverable torque; the
# mean random-walks freely inside the plateau, and exploration noise is
# invisible there too -- entropy becomes free, std ran away 0.43 -> 3.73, and
# the executed torque degenerates into bang-bang. That is the vibration, and
# the instant backward fall seen in mc_mujoco position mode.
#
# Why 1.0 specifically: at 1.0 one raw action unit is exactly this actuator's
# effort_limit/stiffness, which is also the half-width of the feasibility
# window at ratio 1. So the projection window is ``ratio`` action units wide
# and comparable to the exploration std -- the policy explores mostly-feasible
# actions and sees a real gradient at the boundary. At 7.0 the same window is
# 1/7 of an action unit: the projection would bite on nearly every sample and
# bring the plateau straight back, worse than no projection at all.
#
# Reachability, the reason 7.0 was picked in the first place, is handled by
# init_std, not by the scale: PPO moves the policy mean at a rate proportional
# to sigma in raw-action space, while the excursion to reach in raw units is
# proportional to 1/scale, so time-to-reach goes as 1/(sigma * scale).
# 0.4286 * 7.0 = 3.0, so init_std = 3.0 at scale 1.0 reproduces the drift rate
# that made 7.0 work -- and lands on the same 0.021 rad 1-sigma physical design
# target this file has carried across every previous rescale. See rl_cfg.py.
#
# No reward weight moves with this change: action_rate_l2 and
# stance_action_acc_l2 (the two whose weights had to track scale^2) are popped
# in env_cfgs.py, and their replacement action_jerk_l2 multiplies by scale^2
# internally, so it is already in physical units.
# Global gains on top of the formula. These are pure reparametrisations -- a
# common factor on the scale with init_std divided by the same factor leaves
# every physical quantity identical (exploration amplitude, drift rate, torque)
# and changes only the numeric range the network's output layer has to reach.
# That range is not free: this file already records that at scale 0.0105 the raw
# actions needed for a 0.3-0.4 rad hip excursion were 30-70 and "essentially
# unreachable through normal gradient progress". The bare formula puts a modest
# 0.2 rad hip excursion at 143 raw units -- deeper into the regime that was
# measured to fail. The gain exists to avoid re-running that experiment.
#
# 35.0 is not tuned, it is read off the run that works (2026-07-29_01-13-36,
# leg scale 7 * effort_limit/stiffness): 35 * e/(kp + kd/dt) reproduces its hip,
# knee and crotch scales to 1.00, because kd/dt is exactly 4*kp on those
# actuators. So the network's output range is unchanged from a configuration
# known to train, and the only substantive move is the ankles at 0.71 -- the
# correction this formula exists for, since they were the joints the old
# effort/stiffness form over-credited (kd/dt is 6*kp there, not 4).
# Back to 7.0 (2026-07-30). Run 2026-07-29_01-13-36 is the only configuration of
# this robot observed to walk, and its leg scale was 7 * effort_limit/stiffness.
# Everything this session did to the scale was either inert (35 * e/(kp+kd/dt) is
# identically 7 * e/kp, since kd/dt = 4*kp on the legs) or a deviation from the
# one baseline that works. The measured defect of that run was not its scale: it
# was that 88-96% of its leg commands sat outside the executable set, which the
# feasibility projection addresses without touching the action space.
# Back to 1.5, policy 0's multiplier -- the policy that actually walked on the
# robot.
#
# The multiplier, not that run's absolute values: policy 0's invariant is "one
# action unit = 1.5x saturation", and it must track effort_limit. The knee was
# at 100 and is now 70, so its scale goes 0.0075 -> 0.00525. Intended, not
# drift. The five other leg joints are unchanged.
#
# Why go back at all: on hardware this scale is what "conservative" means. It is
# also the reason the raw-torque penalty existed -- the campaign that built it
# started by raising the scale 4.67x. Taking the small scale means the torque
# problem is much smaller to begin with, and raw_torque_peak is not needed with
# it.
_LEG_SCALE_MULTIPLIER = 1.5
for k in list(RHPS1_ACTION_SCALE):
  if not any(
    tok in k for tok in ("CHEST", "SHOULDER", "ELBOW", "WRIST", "HAND", "HEAD")
  ):
    RHPS1_ACTION_SCALE[k] *= _LEG_SCALE_MULTIPLIER

if __name__ == "__main__":
  from mjlab.entity.entity import Entity

  robot = Entity(get_rhps1_robot_cfg())
  mujoco.viewer.launch(robot.spec.compile())
