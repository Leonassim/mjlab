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
    add_site_if_missing(body, f"{prefix}_foot", (0.0, 0.0, -0.08), (1, 0, 0, 1))
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
# Actuator config (fill with real values for RHPS1).
##

# Example: adjust joint patterns, stiffness, damping, effort_limit, armature.
RHPS1_ACTUATOR_CROTCH_Y = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r".*_CROTCH_Y",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=35.0,
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

# effort_limit below: real per-side ELMO current-limit-derived torque bound, NOT the
# previous flat placeholder (100.0, was ~= URDF's effort="108" for L/R_KNEE_P, rounded).
# tau_max = eta * N * Kt * i_limit, using PL[1] (peak current, valid <= PL[2]=8s before the
# drive derates to CL[1]) rather than the continuous CL[1] bound -- deliberately optimistic
# for now (a training episode is assumed short enough vs. 8s); the flat clamp still can't
# express the drive's actual derate-after-8s behavior, only its magnitude. eta=0.77 is a
# calibrated (not measured) estimate: see RHPS1_gains/pdgains/PositionControlSimulation.ipynb
# "eta calibration" section for the two independent cross-checks it's based on.
#   N=210, Kt=0.101 Nm/Arms (same both sides, real, RHPS1_gains/FromRealRobot/drive_gains_map.csv)
#   L: PL[1]=6.79A -> tau_max_peak = 0.77*210*0.101*6.79 ~= 110.9 Nm (was flat 100.0)
#   R: PL[1]=4.24A -> tau_max_peak = 0.77*210*0.101*4.24 ~=  69.3 Nm (was flat 100.0 -- the
#      old value EXCEEDED this joint's real peak capability, let alone its continuous one)
RHPS1_ACTUATOR_KNEE_L = FiniteDifferencePdActuatorCfg(
  target_names_expr=(r"L_KNEE_P",),
  stiffness=20000.0,
  damping=400.0,
  effort_limit=110.9,
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
  effort_limit=69.3,
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
  pos=(0.0, 0.0, 0.850698),
  joint_pos={
    "R_CROTCH_Y": 0.010533,
    "R_CROTCH_R": -0.028787,
    "R_CROTCH_P": -0.196399,
    "R_KNEE_P": 0.487266,
    "R_ANKLE_R": 0.030652,
    "R_ANKLE_P": -0.290867,
    "L_CROTCH_Y": -0.010533,
    "L_CROTCH_R": 0.028787,
    "L_CROTCH_P": -0.196399,
    "L_KNEE_P": 0.487266,
    "L_ANKLE_R": -0.030652,
    "L_ANKLE_P": -0.290867,
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


RHPS1_ACTION_SCALE: dict[str, float] = {}
for a in RHPS1_ARTICULATION.actuators:
  assert isinstance(a, FiniteDifferencePdActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
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
  "CHEST_Y",
  "CHEST_P",
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
):
  RHPS1_ACTION_SCALE[name] = upper_scale

# With the policy's unit-std Gaussian exploration, this scale IS the
# exploration amplitude in joint space -- never drive it with a curriculum,
# unlike reward-shaping terms, since it redefines what a raw network output
# means mid-training. Deployment note: the controller yaml action_scale
# must match the training that produced the deployed ONNX.
_LEG_SCALE_MULTIPLIER = 1.5
for k in list(RHPS1_ACTION_SCALE):
  if k not in [
    n
    for n in RHPS1_ACTION_SCALE
    if any(tok in k for tok in ("CHEST", "SHOULDER", "ELBOW", "WRIST", "HAND", "HEAD"))
  ]:
    RHPS1_ACTION_SCALE[k] *= _LEG_SCALE_MULTIPLIER

if __name__ == "__main__":
  from mjlab.entity.entity import Entity

  robot = Entity(get_rhps1_robot_cfg())
  mujoco.viewer.launch(robot.spec.compile())
