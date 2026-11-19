"""RHPS1_STAIRS=1: stairs on part of the terrain, the BWC copy unchanged on flat tiles.

Flat tiles keep every term of the copy; the reference is only re-anchored on the
tile. Stair tiles (a pyramid of N steps: climb to the platform, or descend from it)
drop the reference: the command is forward until the robot reaches the far level,
then zero, and the rewards are progress, arrival, no riser kicks and balance at
every stance (capture point inside the supported sole area; CoM inside is a bonus).
The policy sees the terrain through a yaw-aligned height map.

The ground is one plane under the grid and the steps are boxes on it: the copy
fell on flat box tiles (box-box contact) where it stands on the plane it was
trained on (2026-11-06, 3-4 falls in 2 s vs 0).
"""

import dataclasses
import math
import os

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
import mujoco
import numpy as np

from dataclasses import dataclass, field

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .standing_start import StandingStartMotionCommand, StandingStartMotionCommandCfg


def _env(k: str, d: str) -> str:
  return os.environ.get("RHPS1_STAIRS_" + k, d)


TILE = float(_env("TILE", "10.0"))
NSTEPS = int(_env("NSTEPS", "3"))  # risers; 10 s episodes leave no time for more
PLATFORM = float(_env("PLATFORM", "1.6"))
TREADS = tuple(float(x) for x in _env("TREADS", "0.27,0.30").split(","))
H_RANGE = tuple(float(x) for x in _env("H", "0.02,0.18").split(","))
ROWS = int(_env("ROWS", "10"))
FRAC = float(_env("FRAC", "0.5"))
SPEED = float(_env("SPEED", "0.12"))
VEL_TAU = float(_env("VEL_TAU", "0"))  # s; >0: progress reward on the smoothed forward velocity
VEL_STD = float(_env("VEL_STD", "0.12"))
FOOTHOLD_PRE = _env("FOOTHOLD_PRE", "0") == "1"  # foothold penalty also on the landing swing foot
FOOTHOLD_REL = _env("FOOTHOLD_REL", "0") == "1"  # edge drop threshold = half the riser: placement from level 0
# Rays allowed off the tread before the foothold cost starts, in descents (climbs keep 2). Leo, 2026-11-17: in
# descent the foot lands astride the nosing. 2 = unchanged; 1 = cost from one full ray row (3 of 6) at 0.5.
FOOTHOLD_DESC_FREE = float(_env("FOOTHOLD_DESC_FREE", "2"))
STATIC_AIR = float(_env("STATIC_AIR", "0"))  # s; >0: static balance paid only during a swing shorter than this
STANCE_W0 = float(_env("STANCE_W0", "0.20"))  # m, lateral ankle distance of the initial posture
STANCE_TOL = float(_env("STANCE_TOL", "0.03"))  # m, free band around it
T_STEP = float(_env("T_STEP", "2.0"))  # s, minimum time between two landings (Leo: ~2 s per step on stairs)
CAD_RAMP = int(_env("CAD_RAMP", "24000"))  # env steps from this launch to the full cadence cost
SWING_MAX = float(_env("SWING_MAX", "0"))  # s, >0: longer swings lose the cadence pay and pay stairs_long_swing
UP_LOAD = float(_env("UP_LOAD", "0"))  # BW fraction, >0: a foot counts as down only above it for UP_T (taps do not)
UP_T = float(_env("UP_T", "0.1"))
MID_DESC_P = float(_env("MID_DESC_P", "0"))  # share of descents spawned on an inner tread, facing down
PAUSE_P = float(_env("PAUSE_P", "0"))  # per-episode chance of a 1.5-3.5 s command pause mid-walk (stop on two feet)
# Leo 2026-11-14: stick released on the stairs. Per-episode chance of a 2-5 s stop triggered when the pelvis passes a
# random point of the flight (replaces the timed pause): time-drawn pauses landed on the flat approach.
STAIR_PAUSE_P = float(_env("STAIR_PAUSE_P", "0"))
SETTLE_T = float(_env("SETTLE_T", "0"))  # s, >0: the arrival bonus waits until still on two feet (25% BW each) this long
SETTLE_V = float(_env("SETTLE_V", "0.05"))  # m/s, v_ema bound for "settled"
ONE_LEG_OFF = float(_env("ONE_LEG_OFF", "0"))  # s, >0: one-leg termination threshold while the command is off
ARRIVE_LOADED = _env("ARRIVE_LOADED", "0") == "1"  # arrival and pause starts wait for both feet down (10% BW)
STOP_DS = _env("STOP_DS", "0") == "1"  # stairs_stop paid only with both feet carrying UP_LOAD (needs UP_LOAD)
ONE_LEG_T = float(_env("ONE_LEG_T", "0"))  # s, >0: a foot up longer ends the episode with the fall cost (CaT)
CAD_C = float(_env("CAD_C", "0"))  # >0: two-sided cadence, min(gap/T_STEP, 1) - CAD_C per landing (weight > 0)
AIR_WIN = tuple(float(x) for x in _env("AIR_WIN", "0.05,0.6").split(","))  # s, swing time paid by stairs_air
AIR_WIN0 = tuple(float(x) for x in _env("AIR_WIN0", _env("AIR_WIN", "0.05,0.6")).split(","))  # window at launch, ramped to AIR_WIN
CLIMB_FRAC = float(_env("CLIMB_FRAC", "0.5"))  # share of stair spawns that climb
# MODE_FIXED=1: each env keeps its mode, so climbs and descents each get their own curriculum level
# (shared, rl40 settled at 5 cm where climbs arrived 44% and descents 12%: descents stopped learning).
MODE_FIXED = _env("MODE_FIXED", "0") == "1"
# REPLAY_P: share of stair episodes on a random level at or below the env's curriculum level, which
# they leave unchanged (rl40 forgot 0.5 cm steps once the levels moved up: 2/2 at 11000, 0/2 after).
REPLAY_P = float(_env("REPLAY_P", "0"))
LVL_FILE = _env("LVL_FILE", "")  # per-env curriculum levels, saved during training and loaded on resume
SCAN_SIZE = tuple(float(x) for x in _env("SCAN", "1.2,0.6").split(","))
SCAN_RES = float(_env("SCAN_RES", "0.05"))
SCAN_OFFSET = float(_env("SCAN_OFFSET", "0.84"))
SCAN_FWD = float(_env("SCAN_FWD", "0"))  # m, grid centre ahead of the pelvis (0.3: -0.3 to +0.9 m)
SCAN_NOISE = float(_env("SCAN_NOISE", "0.02"))  # actor-scan uniform noise, m; 0 = exact teacher map
BW = 57.64 * 9.81
GAP = tuple(float(x) for x in _env("GAP", "0.15,0.35").split(","))  # climb spawn distance from the first riser, m
# Leo 2026-11-14: approach from flat ground at an angle, place the feet before the first step. Spawn heading
# spread around the stairs axis (+-, deg) and a yaw command (sign, as the direction command) toward that axis.
YAW_SPAWN = math.radians(float(_env("YAW_SPAWN", str(math.degrees(0.15)))))
HEAD_CMD = _env("HEAD_CMD", "0") == "1"
HEAD_DB = math.radians(float(_env("HEAD_DB", "10")))  # no yaw command inside this heading error
# ap2 43000 froze on the run-up: flat ground in the scan had only ever meant "arrived, stand". GAP, YAW_SPAWN and
# DESC_EDGE open from the old spawn (0.15-0.35 m, +-0.15 rad, 0.05-0.5 m) over APPROACH_RAMP env steps.
APPROACH_RAMP = int(_env("APPROACH_RAMP", "0"))
APPROACH_OFF = int(_env("APPROACH_OFF", "0"))  # env steps of the ramp already done (resume)
# ap4 43250 drifted to +-60 deg on the run-up: exp(-e^2/0.3^2) is ~0 past 25 deg, no pull back from 30 deg spawns.
HEADING_STD = float(_env("HEADING_STD", "0.3"))
DESC_EDGE = tuple(float(x) for x in _env("DESC_EDGE", "0.25,0.5").split(","))  # top-start descents: distance to the edge, m
ARRIVE = 0.2  # pelvis past the last edge, m
ARRIVE_DESC = float(_env("ARRIVE_DESC", str(ARRIVE)))  # descents: flat ground walked past the last edge before the stop, m
ARRIVE_FEET = _env("ARRIVE_FEET", "0") == "1"  # arrival also needs both ankles past the last edge
HOLD_S = float(_env("HOLD_S", "1.0"))  # arming hold on stair envs (copy: up to 3 s)
EPISODE_S = float(_env("EPISODE_S", "0"))  # stair-env episode length, 0 = the copy's
HELD_ARRIVAL = _env("HELD_ARRIVAL", "1") == "1"  # curriculum promotes only arrivals without a fall
# rl48b/49 sat at the level where half the climbs fall (one held arrival up, one failure down).
PROMOTE_N = int(_env("PROMOTE_N", "1"))  # consecutive held arrivals at a level before promotion
# LVL_FREEZE=1: levels never move (loaded from LVL_FILE). For distillation: the student's failures must not pull the
# level mix away from the teacher's, the teacher labels the hard levels whatever the student does.
LVL_FREEZE = _env("LVL_FREEZE", "0") == "1"
# rl53 39000: once the knee cost cut the falls the curriculum climbed 3.8 -> 4.9 in 500 its and the falls came back:
# the frontier keeps every env at ~50 %. LVL_MAX caps it (an env past the cap is redrawn uniformly below it), so the
# training mass stays on the heights the robot must master (Leo: 5 cm reliable first).
LVL_MAX = int(_env("LVL_MAX", "-1"))
TEACHER_MASK = _env("TEACHER_MASK", "flat")  # flat: only where the scan is flat; all: whole stair tile
TEACHER = _env("TEACHER", "logs/rsl_rl/rhps1_tracking/2026-11-06_stairs_init/model_13498.pt")
# Full RL (Leo, 2026-11-09): no copy term, the goal reached at the robot's own pace.
RL = _env("RL", "0") == "1"
# Leo 2026-11-14, from-scratch 50 Hz run: flat tiles run the RL task (walk to a goal, stop), not the copy; STAND_P of
# the episodes get no command at all (stand still). Both stay out of the curriculum.
FLAT_RL = _env("FLAT_RL", "0") == "1"
STAND_P = float(_env("STAND_P", "0"))
# User 2026-11-19 (gamepad in mc_mujoco): the stairs policy must also side-step and turn in place to line up with the
# stairs. OMNI_P of the flat-tile episodes get a lateral (vy = +-1) or turn (yaw = +-1) command instead of forward;
# progress pays the lateral metres or OMNI_R x the turned angle (same ratchet and VCAP), then the robot must stop.
OMNI_P = float(_env("OMNI_P", "0"))
OMNI_SIDE_FRAC = float(_env("OMNI_SIDE_FRAC", "0.5"))  # side-step share of the omni episodes, the rest turn
OMNI_SIDE = tuple(float(x) for x in _env("OMNI_SIDE", "0.3,0.8").split(","))  # lateral distance, m
OMNI_TURN = tuple(float(x) for x in _env("OMNI_TURN", "0.5,1.6").split(","))  # turn angle, rad
OSEQ_P = float(_env("OSEQ_P", "0"))  # share of flat tiles running a command sequence (o50, Leo 2026-11-19)
OSEQ_N = tuple(int(x) for x in _env("OSEQ_N", "3,5").split(","))  # segments per episode, then zero until the end
OSEQ_T = tuple(float(x) for x in _env("OSEQ_T", "1.5,4.0").split(","))  # s per segment
OSEQ_ZERO = float(_env("OSEQ_ZERO", "0.25"))  # share of zero segments (stop and restart)
OSEQ_COMBO = float(_env("OSEQ_COMBO", "0.3"))  # share of 2- and 3-axis sign combos among the moving segments
OSEQ_V = tuple(float(x) for x in _env("OSEQ_V", "0.12,0.08,0.08,0.3").split(","))  # fwd, back, side m/s, yaw rad/s
OSEQ_TAU = float(_env("OSEQ_TAU", "0.4"))  # s, body-velocity smoothing for the tracking reward
OSEQ_STD = float(_env("OSEQ_STD", "0.7"))  # tracking kernel width, in units of the per-axis target speed
OMNI_R = float(_env("OMNI_R", "0.4"))  # m per rad of turn in the progress ratchet (VCAP 0.12 -> 0.3 rad/s)
STALL_S = float(_env("STALL_S", "3.0"))  # s without 1 cm of new progress before the stall cost
STALL_DESC_X = float(_env("STALL_DESC_X", "1"))
VMAX = float(_env("VMAX", "0.25"))  # m/s, smoothed; faster costs
# rl1 ran the course at ~0.3 m/s and fell off the far edge: progress beyond VCAP on average is not paid.
VCAP = float(_env("VCAP", "0"))  # m/s, 0 = off
BUCKET = float(_env("BUCKET", "0.1"))  # m of credit, so in-step pelvis surges are still paid
# Copy terms kept in RL mode: the robot's own regularizers, none reads the reference.
_RL_KEEP = ("action_rate_l2", "joint_limit", "self_collisions", "slip_speed", "ankle_vibration", "roll_rate")
# BWC landing limits (rl3 landed 3x harder than slot 20). A landing needs LAND_AIR s airborne first:
# under exploration noise the copy's caps counted every contact chatter and rl4 fell on purpose.
TD_VMAX = float(_env("TD_VMAX", "0.21"))  # m/s, the BWC's fastest landing
IMP_FMAX = float(_env("IMP_FMAX", "729"))  # N, slot 13's p99
LAND_AIR = float(_env("LAND_AIR", "0.08"))
LAND_RAMP = int(_env("LAND_RAMP", "12000"))  # env steps from this launch (the counter survives a resume)
LAND_RAMP_OFF = int(_env("LAND_RAMP_OFF", "0"))  # env steps of the ramp already done (resume)
# rl7 stomped at 2 m/s / 4 kN p90: a squared excess capped at 2x prices every stomp alike. "lin": linear, capped at LAND_CAP x.
LAND_SHAPE = _env("LAND_SHAPE", "sq")
LAND_CAP = float(_env("LAND_CAP", "1.0"))
# rl8 at 3000: knee at its limit 20-70 % of the time, pelvis p95 10-32 deg, hopping (Leo: neutral trunk, slow).
TRQ_FRAC = float(_env("TRQ_FRAC", "0.8"))  # leg torque free up to this fraction of the drive limit
# rl11 walked 0.5 cm gently but froze at 2.4 cm risers: the soft gait lifted its feet 1-2 cm.
CLEAR_H = float(_env("CLEAR_H", "0.06"))  # m, swing sole height over the ground under the ankle
CLEAR_FWD = _env("CLEAR_FWD", "0") == "1"  # swing_clearance only for a foot moving toward the goal
# rl15 dived off steps at 0.6-1.6 m/s under an overspeed cost on a 1 s average; and the climb needs some lean.
VINST = float(_env("VINST", "0.35"))  # m/s, instantaneous pelvis speed allowed
TRUNK_DEAD = math.radians(float(_env("TRUNK_DEAD", "0")))  # tilt allowed for free
FALL_S0 = float(_env("FALL_S0", "1"))  # fall cost fraction at launch, ramped to 1 with the regularizers
FCLEAR_RAMP = _env("FCLEAR_RAMP", "0") == "1"
_TRQ_LIM = {"KNEE": 108.0, "ANKLE_P": 65.0, "ANKLE_R": 45.0, "CROTCH_P": 140.0, "CROTCH_R": 100.0, "CROTCH_Y": 35.0}
# rl48b sat at the clip (knee 12 % of the time at 108, over 70 18 %): a cost that starts below the clip and grows
# linearly, so saturation is never free. TRQ_LIM="KNEE=70": per-joint maxima (the robot's knee reducer is 70).
_TRQ_LIM.update({k: float(v) for k, v in (x.split("=") for x in _env("TRQ_LIM", "").split(",") if x)})
TRQ_SHAPE = _env("TRQ_SHAPE", "sq")  # "lin": linear excess, nonzero slope at the threshold
# rl50 in mc_mujoco: the stance knee pinned at 108 N.m for 0.7-1 s on a step-up, then the robot fell backward.
SAT_T = float(_env("SAT_T", "0"))  # s, >0: a leg joint at >=97 % of its limit this long ends the episode (CaT)
SAT_T0 = float(_env("SAT_T0", "0"))  # s, >SAT_T: threshold at launch, ramped down to SAT_T over SAT_RAMP env steps
# rl51 36000 at 1.4 cm: ~35 cm lunges landing on a 55-60 deg knee, which then lifted the body at 108 N.m for
# 0.5-1 s per step and buckled (forward falls) or pushed the robot back. A saturated torque is flat in the
# flexion, the angle is not. Allowance: KNEE_FLEX deg + KNEE_FLEX_H deg per cm of riser (static study: 55-80 at 18 cm).
KNEE_FLEX = math.radians(float(_env("KNEE_FLEX", "40")))
KNEE_FLEX_H = math.radians(float(_env("KNEE_FLEX_H", "2")))
SAT_RAMP = int(_env("SAT_RAMP", "0"))
# 50 Hz: a policy step spans 8 physics steps and the sampled foot load misses 5-15 ms impact peaks. FPEAK_SUB=1
# keeps the feet sensor's substep history and takes the landing peak over it.
FPEAK_SUB = _env("FPEAK_SUB", "0") == "1"

# Reference terms: meaningless on stairs, kept on flat tiles only.
_FLAT_ONLY = ("contact_schedule", "double_support", "heading_hold", "still_when_reference_still",
              "rest_load_shift", "rest_yaw_vibration", "yaw_momentum_still", "arm_swing",
              "motion_feet_pos", "head_level", "head_stable", "head_yaw_still")
_FLAT_ONLY_TERMINATIONS = ("anchor_pos", "anchor_ori", "ee_body_pos")


@dataclass(kw_only=True)
class GroundPlaneCfg(SubTerrainCfg):
  """Flat tile, no geometry of its own; the first one adds the plane under the whole grid."""

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    del difficulty, rng
    geoms = []
    body = spec.body("terrain")
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in body.geoms):
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01))
      g.rgba[:] = (0.55, 0.55, 0.55, 1.0)
      geoms.append(TerrainGeometry(geom=g))
    return TerrainOutput(origin=np.array((self.size[0] / 2, self.size[1] / 2, 0.0)), geometries=geoms)


@dataclass(kw_only=True)
class StepPyramidCfg(SubTerrainCfg):
  """NSTEPS nested boxes on the plane: risers of step_height, treads of step_width."""

  step_height_range: tuple[float, float]
  step_width: float
  num_steps: int = 4
  platform_width: float = 1.6

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    del rng
    h = self.step_height_range[0] + difficulty * (self.step_height_range[1] - self.step_height_range[0])
    body = spec.body("terrain")
    cx, cy = self.size[0] / 2, self.size[1] / 2
    geoms = []
    for k in range(1, self.num_steps + 1):
      half = self.platform_width / 2 + (self.num_steps - k) * self.step_width
      zt = k * h
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(half, half, zt / 2), pos=(cx, cy, zt / 2))
      shade = 0.35 + 0.5 * k / self.num_steps
      g.rgba[:] = (0.2 * shade, 0.45 * shade, 0.95 * shade, 1.0)
      geoms.append(TerrainGeometry(geom=g))
    return TerrainOutput(origin=np.array((cx, cy, self.num_steps * h)), geometries=geoms)


class _State:
  """Per-env stairs task state, shared by the command, rewards and terminations."""

  def __init__(self, env, kinds: list[str], treads: list[float]):
    dev = env.device
    n = env.num_envs
    terrain = env.scene.terrain
    types = terrain.terrain_types.to(dev)
    kind = torch.tensor([{"flat": 0, "up": 1}[k] for k in kinds], device=dev)
    tread = torch.tensor(treads, device=dev)
    self.kind = kind[types]
    self.pyr = self.kind > 0  # pyramid tiles (stair geometry)
    self.is_stairs = self.pyr | ((self.kind == 0) & RL & FLAT_RL)  # every env run by the RL task
    self.standing = torch.zeros(n, dtype=torch.bool, device=dev)
    self.omni = torch.zeros(n, dtype=torch.long, device=dev)  # 0 forward, 1 side-step, 2 turn (OMNI_P)
    self.odir = torch.ones(n, device=dev)  # +1 left / counter-clockwise
    self.oaxis = torch.zeros(n, 2, device=dev)  # side-step direction (world)
    self.oang = torch.zeros(n, device=dev)  # turned angle this episode, rad (unwrapped)
    self.oyaw = torch.zeros(n, device=dev)  # last yaw, rad
    self.oseq = torch.zeros(n, dtype=torch.bool, device=dev)  # OSEQ_P: flat env running a command sequence
    self.seg_cmd = torch.zeros(n, OSEQ_N[1] + 1, 3, device=dev)  # signs (vx, vy, yaw); the last slot stays zero
    self.seg_end = torch.full((n, OSEQ_N[1] + 1), 10**9, dtype=torch.long, device=dev)  # steps after cmd_on
    self.ocmd = torch.zeros(n, 3, device=dev)  # current sequence command (signs)
    self.ov = torch.zeros(n, 3, device=dev)  # smoothed body-frame (vx, vy, wz)
    self.tread = tread[types]
    self.nsteps = torch.tensor([NSTEPS if t > 0 else 0 for t in treads], device=dev)[types]
    self.center = torch.zeros(n, 2, device=dev)
    self.u = torch.zeros(n, 2, device=dev)
    self.face = torch.zeros(n, 2, device=dev)
    self.goal = torch.zeros(n, device=dev)  # arrival when sign * (s - goal) > 0
    self.sign = torch.ones(n, device=dev)
    self.climb = torch.zeros(n, dtype=torch.bool, device=dev)
    self.mode_climb = torch.rand(n, device=dev) < CLIMB_FRAC
    self.lvl_cur = None
    self.replayed = torch.zeros(n, dtype=torch.bool, device=dev)
    self.arrived = torch.zeros(n, dtype=torch.bool, device=dev)
    self.arrived_now = torch.zeros(n, dtype=torch.bool, device=dev)
    self.progress = torch.zeros(n, device=dev)  # fraction of the stairs crossed
    self.s0 = torch.zeros(n, device=dev)  # spawn position along u
    self.sp_trig = torch.full((n,), float("nan"), device=dev)  # STAIR_PAUSE_P trigger along u, nan = none
    self.sp_dur = torch.zeros(n, dtype=torch.long, device=dev)
    self.cmd_on = torch.zeros(n, dtype=torch.long, device=dev)
    self.hold_end = torch.zeros(n, dtype=torch.long, device=dev)
    self.pause0 = torch.full((n,), 10**9, dtype=torch.long, device=dev)  # command pause [pause0, pause1), steps
    self.pause1 = torch.full((n,), 10**9, dtype=torch.long, device=dev)
    self.paused = torch.zeros(n, dtype=torch.bool, device=dev)
    self.p_on = torch.zeros(n, dtype=torch.bool, device=dev)  # pause started (latched)
    self.settle_t = torch.zeros(n, device=dev)
    self.settled = torch.zeros(n, dtype=torch.bool, device=dev)
    self.settled_now = torch.zeros(n, dtype=torch.bool, device=dev)
    self.prev_com = torch.zeros(n, 3, device=dev)
    self.com_vel = torch.zeros(n, 3, device=dev)
    self.v_ema = torch.zeros(n, 2, device=dev)  # (forward, lateral) in the facing frame
    self.best = torch.zeros(n, device=dev)  # closest approach to the goal this episode, m
    self.gain = torch.zeros(n, device=dev)  # its decrease over the last step
    self.mark = torch.zeros(n, device=dev)
    self.last_gain = torch.zeros(n, dtype=torch.long, device=dev)
    self.credit = torch.zeros(n, device=dev)
    self.air = torch.zeros(n, 2, device=dev)  # time since each foot was last loaded, s
    self.sup = torch.zeros(n, 2, device=dev)  # time each foot has carried UP_LOAD continuously, s
    self.up = torch.zeros(n, 2, device=dev)  # time since each foot was last really standing (UP_LOAD for UP_T), s
    self.load = torch.zeros(n, 2, device=dev)  # normal load on each foot, N
    self.since_land = torch.full((n,), 10.0, device=dev)  # time since the last landing of either foot, s
    self.land_gap = torch.zeros(n, device=dev)  # that time at this step's landing
    self.swing_land = torch.zeros(n, 2, device=dev)  # landing foot's swing time at this step's landing
    # Landing speed = fastest descent over the last 50 ms (11 samples at 200 Hz, 3 at 50 Hz).
    self.vz_buf = torch.zeros(n, int(round(0.05 / env.step_dt)) + 1, 2, device=dev)
    self.land = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    self.v_land = torch.zeros(n, 2, device=dev)
    self.f_t = torch.ones(n, 2, device=dev)
    self.f_peak = torch.zeros(n, 2, device=dev)
    self.f_close = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    names = env.scene["stairs_feet"].primary_names
    self.foot_ids = [env.scene["robot"].body_names.index(b) for b in names]

  def spawn(self, env, ids: torch.Tensor):
    """Root pose for stair envs: at the bottom facing the stairs, or on top facing the edge."""
    dev = env.device
    k = len(ids)
    origin = env.scene.env_origins[ids]
    c = origin[:, :2]
    ax = torch.randint(0, 4, (k,), device=dev)
    ang = ax.float() * (math.pi / 2)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    climb = self.mode_climb[ids] if MODE_FIXED else torch.rand(k, device=dev) < CLIMB_FRAC
    span = PLATFORM / 2 + (self.nsteps[ids] - 1) * self.tread[ids]  # outer riser
    r = 1.0
    if APPROACH_RAMP > 0:
      c0 = getattr(env, "_ap_c0", None)
      if c0 is None:
        c0 = env._ap_c0 = env.common_step_counter
      r = min(1.0, (env.common_step_counter - c0 + APPROACH_OFF) / APPROACH_RAMP)
    g0, g1 = 0.15 + (GAP[0] - 0.15) * r, 0.35 + (GAP[1] - 0.35) * r
    outside = span + g0 + torch.rand(k, device=dev) * (g1 - g0)
    e0, e1 = 0.05 + (DESC_EDGE[0] - 0.05) * r, 0.5 + (DESC_EDGE[1] - 0.5) * r  # ramped from rl54's 0.05-0.5 m
    inside = PLATFORM / 2 - e1 + torch.rand(k, device=dev) * (e1 - e0)
    # climb: on the plane facing the center; descend: on the platform facing out.
    from_out = climb
    s0 = torch.where(from_out, outside, inside)
    face = torch.where(from_out[:, None], -u, u)
    lat = torch.stack([-u[:, 1], u[:, 0]], 1) * (torch.rand(k, 1, device=dev) - 0.5) * 0.6
    xy = c + u * s0[:, None] + lat
    ground = torch.where(from_out, torch.zeros_like(s0), origin[:, 2])
    if MID_DESC_P > 0:
      # rl39 froze on the top platform from 4 cm on: spawn on tread j (1..n-1 below the platform) to practise stepping down.
      n = self.nsteps[ids]
      mid = ~climb & (torch.rand(k, device=dev) < MID_DESC_P) & (n > 1)
      j = 1 + (torch.rand(k, device=dev) * (n - 1).clamp(min=1).float()).long()
      j = torch.minimum(j, (n - 1).clamp(min=1))
      s_mid = PLATFORM / 2 + (j.float() - 0.5) * self.tread[ids]
      g_mid = origin[:, 2] * (1.0 - j.float() / n.float())
      s0 = torch.where(mid, s_mid, s0)
      ground = torch.where(mid, g_mid, ground)
      xy = c + u * s0[:, None] + lat
    self.center[ids] = c
    self.u[ids] = u
    self.face[ids] = face
    self.goal[ids] = torch.where(from_out, PLATFORM / 2 - ARRIVE, span + ARRIVE_DESC)
    self.sign[ids] = torch.where(from_out, -1.0, 1.0)
    self.climb[ids] = climb
    self.arrived[ids] = False
    self.settle_t[ids] = 0.0
    self.p_on[ids] = False
    self.settled[ids] = False
    self.progress[ids] = 0.0
    self.s0[ids] = s0
    yaw = torch.atan2(face[:, 1], face[:, 0]) + (torch.rand(k, device=dev) * 2 - 1) * (0.15 + (YAW_SPAWN - 0.15) * r)
    self.omni[ids] = 0
    if OMNI_P > 0:
      om = (self.kind[ids] == 0) & (torch.rand(k, device=dev) < OMNI_P)
      side = torch.rand(k, device=dev) < OMNI_SIDE_FRAC
      mode = torch.where(om, torch.where(side, 1, 2), 0)
      odir = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
      hf = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)  # actual heading: the side-step keeps it
      ax = torch.stack([-hf[:, 1], hf[:, 0]], 1) * odir[:, None]
      dist = OMNI_SIDE[0] + torch.rand(k, device=dev) * (OMNI_SIDE[1] - OMNI_SIDE[0])
      ang = OMNI_TURN[0] + torch.rand(k, device=dev) * (OMNI_TURN[1] - OMNI_TURN[0])
      s_side = ((xy - c) * ax).sum(-1)
      g = torch.where(mode == 1, s_side + dist, OMNI_R * ang)
      self.omni[ids] = mode
      self.odir[ids] = odir
      self.oaxis[ids] = ax
      self.oang[ids] = 0.0
      self.oyaw[ids] = yaw
      self.goal[ids] = torch.where(om, g, self.goal[ids])
      self.sign[ids] = torch.where(om, torch.ones_like(odir), self.sign[ids])
      self.face[ids] = torch.where(om[:, None], hf, self.face[ids])
    self.oseq[ids] = False
    if OSEQ_P > 0:
      self._seq_spawn(env, ids)
    return xy, ground, yaw

  def _seq_spawn(self, env, ids: torch.Tensor):
    """Flat tiles: OSEQ_N[0]-OSEQ_N[1] command segments of OSEQ_T s, drawn among zero, the six single axes and
    2-3 axis sign combos (tardis: ~70 % single or zero, ~30 % combos: the joystick can send any of the 27 states)."""
    dev = env.device
    k = len(ids)
    m = OSEQ_N[1]
    sq = (self.kind[ids] == 0) & (torch.rand(k, device=dev) < OSEQ_P)
    nseg = torch.randint(OSEQ_N[0], m + 1, (k,), device=dev)
    single = torch.zeros(k, m, 3, device=dev)
    ax = torch.randint(0, 3, (k, m), device=dev)
    sgn = torch.where(torch.rand(k, m, device=dev) < 0.5, -1.0, 1.0)
    single.scatter_(2, ax[..., None], sgn[..., None])
    combo = torch.where(torch.rand(k, m, 3, device=dev) < 0.5, -1.0, 1.0)
    drop = torch.randint(0, 3, (k, m), device=dev)
    keep3 = torch.rand(k, m, device=dev) < 0.5
    combo = combo * torch.where(keep3[..., None], torch.ones_like(combo),
                                1.0 - torch.nn.functional.one_hot(drop, 3).float())
    r = torch.rand(k, m, device=dev)
    cmd = torch.where((r < OSEQ_COMBO)[..., None], combo, single)
    cmd = cmd * (torch.rand(k, m, device=dev) >= OSEQ_ZERO)[..., None]
    cmd = cmd * (torch.arange(m, device=dev)[None] < nseg[:, None])[..., None]
    dur = ((OSEQ_T[0] + torch.rand(k, m, device=dev) * (OSEQ_T[1] - OSEQ_T[0])) / env.step_dt).long()
    end = dur.cumsum(1)
    last = end.gather(1, (nseg - 1)[:, None])
    end = torch.where(torch.arange(m, device=dev)[None] < nseg[:, None], end, last)  # unused: end with the last
    self.seg_cmd[ids, :m] = cmd
    self.seg_cmd[ids, m] = 0.0
    self.seg_end[ids, :m] = end
    self.seg_end[ids, m] = 10**9
    self.oseq[ids] = sq
    self.ov[ids] = 0.0
    self.goal[ids] = torch.where(sq, torch.full_like(self.goal[ids], 1e6), self.goal[ids])
    self.sign[ids] = torch.where(sq, torch.ones_like(self.sign[ids]), self.sign[ids])

  def update(self, env):
    robot = env.scene["robot"]
    p = robot.data.root_link_pos_w[:, :2]
    s = ((p - self.center) * self.u).sum(-1)
    if OMNI_P > 0:
      q = robot.data.root_link_quat_w
      yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      dy = torch.atan2(torch.sin(yaw - self.oyaw), torch.cos(yaw - self.oyaw))
      self.oang = torch.where(env.episode_length_buf <= 1, torch.zeros_like(dy), self.oang + dy)
      self.oyaw = yaw
      turn = self.omni == 2
      s = torch.where(self.omni == 1, ((p - self.center) * self.oaxis).sum(-1), s)
      s = torch.where(turn, OMNI_R * self.odir * self.oang, s)
      self.face = torch.where(turn[:, None], torch.stack([torch.cos(yaw), torch.sin(yaw)], 1), self.face)
    seq_zero = torch.zeros_like(self.oseq)
    if OSEQ_P > 0:
      ep = env.episode_length_buf
      t = ep - self.cmd_on
      # unused slots end with the last real one, then the extra slot is zero until the end
      idx = (t[:, None] >= self.seg_end).sum(-1).clamp(max=OSEQ_N[1])
      cur = self.seg_cmd.gather(1, idx[:, None, None].expand(-1, 1, 3))[:, 0]
      self.ocmd = torch.where((self.oseq & (t >= 0))[:, None], cur, torch.zeros_like(cur))
      seq_zero = self.oseq & (t >= 0) & (self.ocmd == 0).all(-1)
      q = robot.data.root_link_quat_w
      yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      hf = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
      self.face = torch.where(self.oseq[:, None], hf, self.face)
      v = robot.data.root_link_lin_vel_w[:, :2]
      vb = torch.stack([(v * hf).sum(-1), -v[:, 0] * hf[:, 1] + v[:, 1] * hf[:, 0],
                        robot.data.root_link_ang_vel_w[:, 2]], 1)
      a = min(1.0, env.step_dt / OSEQ_TAU)
      self.ov = torch.where((ep <= 1)[:, None], torch.zeros_like(vb), self.ov + a * (vb - self.ov))
    span = PLATFORM / 2 + (self.nsteps - 1) * self.tread
    start = torch.where(self.sign < 0, span, PLATFORM / 2)
    end = torch.where(self.sign < 0, PLATFORM / 2, span)
    den = end - start
    frac = ((s - start) / torch.where(den.abs() > 1e-6, den, torch.ones_like(den))).clamp(0.0, 1.0)
    self.progress = torch.where(self.pyr, torch.maximum(self.progress, frac), self.progress)
    if STAIR_PAUSE_P > 0:
      ep = env.episode_length_buf
      trig = ~torch.isnan(self.sp_trig) & (self.sign * (s - self.sp_trig) > 0) & (ep >= self.cmd_on) & ~self.arrived
      trig = trig & (self.omni == 0)
      self.pause0 = torch.where(trig, ep, self.pause0)
      self.pause1 = torch.where(trig, ep + self.sp_dur, self.pause1)
      self.sp_trig = torch.where(trig, torch.full_like(self.sp_trig, float("nan")), self.sp_trig)
    win = (env.episode_length_buf >= self.pause0) & (env.episode_length_buf < self.pause1)
    vm = getattr(env, "_vision", None)
    if vm is not None and vm.gate_on:  # RHPS1_VIS_GATE: the controller pauses the command on a doubtful map
      win = win | vm.gate_pause(int(env.common_step_counter))
    win = win | seq_zero  # a zero segment is a pause: the stop rewards and the ARRIVE_LOADED latch apply
    if ARRIVE_LOADED:
      # The command used to drop mid-swing and the policy froze that swing (front foot up after the goal).
      down = (self.load > 0.1 * BW).all(-1)
      self.p_on = (self.p_on | (win & down)) & win
      self.paused = self.p_on
    else:
      self.paused = win
    on = (env.episode_length_buf >= self.cmd_on) & ~self.paused
    hit = self.is_stairs & on & (self.sign * (s - self.goal) > 0) & ~self.arrived & ~self.oseq
    if ARRIVE_FEET:
      # rl14 "arrived" diving off the last step, feet still on the tread: both feet past the last edge.
      sf = ((robot.data.body_link_pos_w[:, self.foot_ids, :2] - self.center[:, None]) * self.u[:, None]).sum(-1)
      edge = torch.where(self.sign < 0, PLATFORM / 2, span)
      hit = hit & ((self.sign[:, None] * (sf - edge[:, None]) > 0).all(-1) | (self.omni > 0))
    if ARRIVE_LOADED:
      hit = hit & (self.load > 0.1 * BW).all(-1)
    self.arrived_now = hit
    self.arrived |= hit
    com = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]
    fresh = (env.episode_length_buf <= 1)[:, None]
    self.com_vel = torch.where(fresh, torch.zeros_like(com), (com - self.prev_com) / env.step_dt)
    self.prev_com = com.clone()
    # In-step speed oscillation made standing still out-earn the gait on the raw velocity.
    v = robot.data.root_link_lin_vel_w[:, :2]
    f = self.face
    vf = torch.stack([(v * f).sum(-1), -v[:, 0] * f[:, 1] + v[:, 1] * f[:, 0]], 1)
    a = min(1.0, env.step_dt / VEL_TAU) if VEL_TAU > 0 else 1.0
    self.v_ema = torch.where(fresh, torch.zeros_like(vf), self.v_ema + a * (vf - self.v_ema))
    ep = env.episode_length_buf
    fr = fresh[:, 0]
    _, _, _, _, fn = _feet(env)
    fl = fn.view(fn.shape[0], 2, -1).sum(-1)
    ld = fl > 10.0
    vz = robot.data.body_link_lin_vel_w[:, self.foot_ids, 2]
    self.vz_buf = torch.where(fresh[:, :, None], vz[:, None].expand_as(self.vz_buf),
                              torch.cat([self.vz_buf[:, 1:], vz[:, None]], 1))
    self.land = ld & (self.air >= LAND_AIR) & ~fresh
    self.v_land = (-self.vz_buf.min(1).values).clamp(min=0.0)
    t = torch.where(self.land, torch.zeros_like(self.f_t), (self.f_t + env.step_dt).clamp(max=1.0))
    fpk = fl
    if FPEAK_SUB:
      h = env.scene["stairs_feet"].data.force_history  # [B, 2 feet x slots, substeps, 3], global frame
      fpk = torch.maximum(fl, h.norm(dim=-1).view(h.shape[0], 2, -1, h.shape[2]).sum(2).max(-1).values)
    self.f_peak = torch.where(self.land, fpk, torch.maximum(self.f_peak, fpk * (t < 0.06)))
    self.f_close = (self.f_t < 0.06) & (t >= 0.06)
    self.f_t = torch.where(fresh, torch.ones_like(t), t)
    self.swing_land = torch.where(self.land, self.air, torch.zeros_like(self.air))
    self.load = fl
    self.air = torch.where(ld | fresh, torch.zeros_like(self.air), self.air + env.step_dt)
    if UP_LOAD > 0:
      # rl27 det held a foot 7 cm up after arrival; under action noise it tapped the ground, resetting air.
      self.sup = torch.where((fl > UP_LOAD * BW) & ~fresh, self.sup + env.step_dt, torch.zeros_like(self.sup))
      self.up = torch.where((self.sup >= UP_T) | fresh, torch.zeros_like(self.up), self.up + env.step_dt)
    else:
      self.up = self.air
    anyl = self.land.any(-1)
    self.land_gap = self.since_land.clone()
    self.since_land = torch.where(fr, torch.full_like(self.since_land, 10.0),
                                  torch.where(anyl, torch.zeros_like(self.since_land), self.since_land + env.step_dt))
    rem = (-self.sign * (s - self.goal)).clamp(min=0.0)
    best = torch.where(on, torch.minimum(self.best, rem), self.best)
    self.gain = torch.where(fr, torch.zeros_like(rem), self.best - best)
    self.best = torch.where(fr, rem, best)
    if VCAP > 0:
      c = torch.where(fr, torch.zeros_like(rem), (self.credit + VCAP * env.step_dt * on).clamp(max=BUCKET))
      self.gain = torch.minimum(self.gain, c)
      self.credit = c - self.gain
    if OSEQ_P > 0:
      self.gain = torch.where(self.oseq, torch.zeros_like(self.gain), self.gain)  # paid by stairs_otrack instead
    moved = (self.best < self.mark - 0.01) | self.oseq  # no stall cost on sequence envs
    self.mark = torch.where(fr | moved, self.best, self.mark)
    self.last_gain = torch.where(fr | moved | ~on | self.arrived, ep, self.last_gain)
    if SETTLE_T > 0:
      calm = self.arrived & (self.load.min(-1).values > 0.25 * BW) & (self.v_ema.norm(dim=-1) < SETTLE_V)
      self.settle_t = torch.where(calm & ~fr, self.settle_t + env.step_dt, torch.zeros_like(self.settle_t))
      self.settled_now = (self.settle_t >= SETTLE_T) & ~self.settled
      self.settled |= self.settled_now


class StairsMotionCommand(StandingStartMotionCommand):
  """The copy's command, re-anchored on its tile; stair envs spawn and walk on their own."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._off = torch.zeros(self.num_envs, 3, device=self.device)
    self._in_reset = False
    env._stairs = _State(env, cfg.stairs_kinds, cfg.stairs_treads)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return super().body_pos_w + self._off[:, None, :]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return super().anchor_pos_w + self._off

  def reset(self, env_ids):
    self._in_reset = True
    try:
      return super().reset(env_ids)
    finally:
      self._in_reset = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    hold = getattr(self._env, "_action_hold_steps", None)
    saved = hold.clone() if hold is not None else None
    super()._resample_command(env_ids)
    st = self._env._stairs
    sid = env_ids[st.is_stairs[env_ids]]
    if len(sid) == 0:
      return
    hold = getattr(self._env, "_action_hold_steps", None)
    if hold is None:
      hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self._env._action_hold_steps = hold
    if self._in_reset:
      hmax = HOLD_S
      dt = self._env.step_dt
      h = torch.randint(0, max(int(hmax / dt), 1), (len(sid),), device=self.device)
      hold[sid] = h
      st.hold_end[sid] = h
      st.cmd_on[sid] = h + (torch.rand(len(sid), device=self.device) * 0.6 / dt + 0.2 / dt).long()
      stand = torch.rand(len(sid), device=self.device) < STAND_P
      st.standing[sid] = stand
      st.cmd_on[sid] = torch.where(stand, torch.full_like(st.cmd_on[sid], 10**9), st.cmd_on[sid])
      if PAUSE_P > 0:
        # Stop and stand mid-walk, then go on: the only stop rl21-31 saw was the arrival, and they balanced on one leg.
        k = len(sid)
        p0 = st.cmd_on[sid] + ((1.5 + 4.5 * torch.rand(k, device=self.device)) / dt).long()
        p1 = p0 + ((1.5 + 2.0 * torch.rand(k, device=self.device)) / dt).long()
        yes = torch.rand(k, device=self.device) < PAUSE_P
        st.pause0[sid] = torch.where(yes, p0, torch.full_like(p0, 10**9))
        st.pause1[sid] = torch.where(yes, p1, torch.full_like(p1, 10**9))
      if STAIR_PAUSE_P > 0:
        k = len(sid)
        span = PLATFORM / 2 + (st.nsteps[sid] - 1) * st.tread[sid]
        lo = torch.where(st.climb[sid], torch.full_like(span, PLATFORM / 2), st.s0[sid].clamp(min=PLATFORM / 2))
        trig = lo + torch.rand(k, device=self.device) * (span - lo).clamp(min=0.0)
        yes = torch.rand(k, device=self.device) < STAIR_PAUSE_P
        st.sp_trig[sid] = torch.where(yes, trig, torch.full_like(trig, float("nan")))
        st.sp_dur[sid] = ((2.0 + 3.0 * torch.rand(k, device=self.device)) / dt).long()
        st.pause0[sid] = torch.where(yes, torch.full_like(st.pause0[sid], 10**9), st.pause0[sid])
        st.pause1[sid] = torch.where(yes, torch.full_like(st.pause1[sid], 10**9), st.pause1[sid])
    elif saved is not None:
      hold[sid] = saved[sid]  # a clip wrap must not re-hold a walking stair env

  def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel,
                                    root_ang_vel, joint_pos, joint_vel) -> None:
    st = self._env._stairs
    stairs = st.is_stairs[env_ids]
    root_pos = root_pos.clone()
    flat = ~stairs
    if flat.any() and _env("ANCHOR", "1") == "1":
      fid = env_ids[flat]
      origin = self._env.scene.env_origins[fid]
      self._off[fid, :2] += origin[:, :2] - root_pos[flat, :2]
      root_pos[flat, :2] = origin[:, :2]
    if stairs.any():
      if not self._in_reset:
        keep = ~stairs
        if not keep.any():
          return
        env_ids, root_pos, root_ori = env_ids[keep], root_pos[keep], root_ori[keep]
        root_lin_vel, root_ang_vel = root_lin_vel[keep], root_ang_vel[keep]
        joint_pos, joint_vel = joint_pos[keep], joint_vel[keep]
      else:
        from mjlab.utils.lab_api.math import quat_from_euler_xyz

        sid = env_ids[stairs]
        xy, ground, yaw = st.spawn(self._env, sid)
        robot = self._env.scene["robot"]
        root_pos[stairs, :2] = xy
        root_pos[stairs, 2] = ground + robot.data.default_root_state[sid, 2] + 0.005
        z = torch.zeros_like(yaw)
        root_ori = root_ori.clone()
        root_ori[stairs] = quat_from_euler_xyz(z, z, yaw)
        root_lin_vel = root_lin_vel.clone(); root_lin_vel[stairs] = 0.0
        root_ang_vel = root_ang_vel.clone(); root_ang_vel[stairs] = 0.0
        joint_pos = joint_pos.clone(); joint_pos[stairs] = robot.data.default_joint_pos[sid]
        joint_vel = joint_vel.clone(); joint_vel[stairs] = 0.0
    super()._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  def _update_command(self, env_ids=None):
    super()._update_command(env_ids)
    self._env._stairs.update(self._env)


@dataclass(kw_only=True)
class StairsMotionCommandCfg(StandingStartMotionCommandCfg):
  stairs_kinds: list = field(default_factory=list)
  stairs_treads: list = field(default_factory=list)

  def build(self, env) -> StairsMotionCommand:
    return StairsMotionCommand(self, env)


# ---------------------------------------------------------------------------
# Command, masks
# ---------------------------------------------------------------------------


def _command_wrapper(orig):
  def cmd(env, command_name: str):
    u = orig(env, command_name)
    st = getattr(env, "_stairs", None)
    if st is None:
      return u
    on = ~st.arrived & ~st.paused & (env.episode_length_buf >= st.cmd_on)
    forced = torch.zeros_like(u)
    forced[:, 0] = (on & (st.omni == 0)).to(u.dtype)
    forced[:, 1] = (on & (st.omni == 1)).to(u.dtype) * st.odir
    forced[:, 2] = (on & (st.omni == 2)).to(u.dtype) * st.odir
    if OSEQ_P > 0:
      forced = torch.where(st.oseq[:, None], st.ocmd.to(u.dtype) * on[:, None].to(u.dtype), forced)
    if HEAD_CMD:
      q = env.scene["robot"].data.root_link_quat_w
      yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      d = torch.atan2(st.face[:, 1], st.face[:, 0]) - yaw
      e = torch.atan2(torch.sin(d), torch.cos(d))
      forced[:, 2] = (torch.sign(e) * (e.abs() > HEAD_DB) * on).to(u.dtype)
    return torch.where(st.is_stairs[:, None], forced, u)

  cmd.__name__ = orig.__name__
  return cmd


def _flat_only(func):
  def f(env, *a, **kw):
    r = func(env, *a, **kw)
    st = getattr(env, "_stairs", None)
    if st is None:
      return r
    keep = ~st.is_stairs
    return r & keep if r.dtype == torch.bool else r * keep.to(r.dtype)

  f.__name__ = getattr(func, "__name__", "flat_only")
  return f


def _zeroed(func):
  def f(env, *a, **kw):
    return torch.zeros_like(func(env, *a, **kw))

  f.__name__ = getattr(func, "__name__", "zeroed")
  return f


def _stairs_only(r: torch.Tensor, env) -> torch.Tensor:
  return r * env._stairs.is_stairs.to(r.dtype)


# ---------------------------------------------------------------------------
# Rewards (stair envs only)
# ---------------------------------------------------------------------------


def stairs_lin_vel(env, std: float) -> torch.Tensor:
  st = env._stairs
  robot = env.scene["robot"]
  v = robot.data.root_link_lin_vel_w[:, :2]
  on = (~st.arrived & ~st.paused & (env.episode_length_buf >= st.cmd_on)).float()
  fwd = (v * st.face).sum(-1)
  lat = v[:, 0] * -st.face[:, 1] + v[:, 1] * st.face[:, 0]
  e = (fwd - on * SPEED) ** 2 + lat**2
  if VEL_TAU > 0:
    # Progress: slot 20's smoothed error is as large as SPEED, so a kernel paid standing nearly as much.
    go = (st.v_ema[:, 0] / SPEED).clamp(0.0, 1.0)
    stop = torch.exp(-(st.v_ema**2).sum(-1) / std**2)
    return _stairs_only(torch.where(on > 0, go, stop), env)
  return _stairs_only(torch.exp(-e / std**2), env)


def stairs_progress(env) -> torch.Tensor:
  """Ratchet on the distance left: each new closest approach is paid once, at any pace
  (weight x metres over the episode)."""
  return _stairs_only(env._stairs.gain / env.step_dt, env)


def _land_ramp(env) -> float:
  c0 = getattr(env, "_land_c0", None)
  if c0 is None:
    c0 = env._land_c0 = env.common_step_counter
  return min(1.0, (env.common_step_counter - c0 + LAND_RAMP_OFF) / LAND_RAMP) if LAND_RAMP > 0 else 1.0


def _land_cost(e: torch.Tensor) -> torch.Tensor:
  return e if LAND_SHAPE == "lin" else e**2


def stairs_stance(env) -> torch.Tensor:
  """Lateral ankle distance (pelvis yaw frame) away from the initial posture's, beyond STANCE_TOL, per 5 cm,
  squared (ramped). Leo saw the feet planted very wide."""
  robot = env.scene["robot"]
  st = env._stairs
  d = robot.data.body_link_pos_w[:, st.foot_ids[0], :2] - robot.data.body_link_pos_w[:, st.foot_ids[1], :2]
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  w = (-d[:, 0] * torch.sin(yaw) + d[:, 1] * torch.cos(yaw)).abs()
  e = torch.relu((w - STANCE_W0).abs() - STANCE_TOL) / 0.05
  return _stairs_only(e**2, env) * _land_ramp(env)


def stairs_cadence(env) -> torch.Tensor:
  """Per landing: shortfall of the time since the previous landing under T_STEP (fraction), ramped from launch.
  Quick steps balance dynamically; on stairs Leo wants ~2 s per step to place the foot."""
  st = env._stairs
  r = min(1.0, (env.common_step_counter - getattr(env, "_land_c0", env.common_step_counter) + LAND_RAMP_OFF) / CAD_RAMP) if CAD_RAMP > 0 else 1.0
  if CAD_C > 0:
    # Penalty only: not landing is the cheapest answer, rl21/22 froze. Two-sided (cf. feet air time): standing
    # earns 0, the rate (g - c) / gap peaks at gap = T_STEP. Pay full from launch, shortfall ramped.
    g = (st.land_gap / T_STEP).clamp(max=1.0) - CAD_C
    on = ~st.arrived & ~st.paused & (env.episode_length_buf >= st.cmd_on)
    # Paid only if the step set a 1 cm progress record: rl23 earned it stepping in place.
    adv = (env.episode_length_buf - st.last_gain).float() * env.step_dt < st.land_gap
    if SWING_MAX > 0:
      # The gap must be spent in double support: rl25 held one leg up 2 s for it.
      adv = adv & (st.swing_land <= SWING_MAX).all(-1)
    e = torch.where(g >= 0, g * (on & adv), g * r)
    return _stairs_only(st.land.any(-1).float() * e / env.step_dt, env)
  e = torch.relu(T_STEP - st.land_gap) / T_STEP
  return _stairs_only(st.land.any(-1).float() * e / env.step_dt * r, env)


def stairs_long_swing(env) -> torch.Tensor:
  """A foot up longer than SWING_MAX, growing per SWING_MAX (ramped): pauses belong in double support."""
  e = (torch.relu(env._stairs.up.max(-1).values - SWING_MAX) / SWING_MAX).clamp(max=3.0)
  return _stairs_only(e * (env.episode_length_buf >= env._stairs.hold_end), env) * _land_ramp(env)


def stairs_touchdown(env) -> torch.Tensor:
  """Per landing: squared excess of the foot's landing speed over the BWC's, capped at 1+LAND_CAP x."""
  st = env._stairs
  e = (torch.relu(st.v_land - TD_VMAX) / TD_VMAX).clamp(max=LAND_CAP)
  return (st.land.float() * _land_cost(e)).sum(-1) / env.step_dt * _land_ramp(env)


def stairs_impact(env) -> torch.Tensor:
  """Per landing: squared excess of the 60 ms peak foot load over IMP_FMAX, capped at 1+LAND_CAP x."""
  st = env._stairs
  e = (torch.relu(st.f_peak - IMP_FMAX) / IMP_FMAX).clamp(max=LAND_CAP)
  return (st.f_close.float() * _land_cost(e)).sum(-1) / env.step_dt * _land_ramp(env)


def _tilt2(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return (2 * (x * z - w * y)) ** 2 + (2 * (y * z + w * x)) ** 2


def stairs_trunk(env) -> torch.Tensor:
  """Pelvis and chest tilt, sin^2 each (ramped)."""
  robot = env.scene["robot"]
  if not hasattr(env, "_st_chest"):
    env._st_chest = list(robot.body_names).index("CHEST_P_LINK")
  q = robot.data.body_link_quat_w
  t2 = torch.stack([_tilt2(robot.data.root_link_quat_w), _tilt2(q[:, env._st_chest])], -1)
  if TRUNK_DEAD > 0:
    t2 = torch.sin(torch.relu(torch.asin(t2.clamp(0, 1).sqrt()) - TRUNK_DEAD)) ** 2
  return t2.sum(-1) * _land_ramp(env)


def stairs_roll(env) -> torch.Tensor:
  """Pelvis and chest roll, sin^2 each, no dead zone (weight RHPS1_STAIRS_W_ROLL < 0). tardis 2026-11-19: s50e sways at
  3.8 deg roll p95 vs slot 18's 2.2, all inside TRUNK_DEAD, so stairs_trunk gave no gradient there."""
  robot = env.scene["robot"]
  if not hasattr(env, "_st_chest"):
    env._st_chest = list(robot.body_names).index("CHEST_P_LINK")

  def r2(q):
    w, x, y, z = q.unbind(-1)
    return (2 * (y * z + w * x)) ** 2

  return (r2(robot.data.root_link_quat_w) + r2(robot.data.body_link_quat_w[:, env._st_chest])) * _land_ramp(env)


def stairs_vz(env) -> torch.Tensor:
  """Pelvis vertical speed squared: climbing needs ~0.2 m/s, hopping 1 m/s (ramped)."""
  return env.scene["robot"].data.root_link_lin_vel_w[:, 2] ** 2 * _land_ramp(env)


def stairs_vinst(env) -> torch.Tensor:
  """Instantaneous pelvis horizontal speed over VINST, squared relative excess (ramped)."""
  v = env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  return (torch.relu(v - VINST) / VINST) ** 2 * _land_ramp(env)


def stairs_torque(env) -> torch.Tensor:
  """Leg torque beyond TRQ_FRAC of the per-joint maximum, normalized to 1 there, squared or linear (ramped)."""
  robot = env.scene["robot"]
  lim = getattr(env, "_st_trq_lim", None)
  if lim is None:
    lim = env._st_trq_lim = torch.tensor(
      [next((v for g, v in _TRQ_LIM.items() if g in n), float("inf")) for n in robot.joint_names], device=env.device)
  r = torch.relu(robot.data.qfrc_actuator.abs() / lim - TRQ_FRAC) / (1 - TRQ_FRAC)
  return (r if TRQ_SHAPE == "lin" else r**2).sum(-1) * _land_ramp(env)


def stairs_knee_flex(env) -> torch.Tensor:
  """Flexion of a loaded knee (> 0.1 BW on its foot) beyond the step's allowance, rad, linear."""
  robot = env.scene["robot"]
  ids = getattr(env, "_st_knee_ids", None)
  if ids is None:
    ids = env._st_knee_ids = [robot.joint_names.index(n) for n in ("L_KNEE_P", "R_KNEE_P")]
  lvl = env.scene.terrain.terrain_levels.float()
  h_cm = 100 * (H_RANGE[0] + lvl / max(ROWS - 1, 1) * (H_RANGE[1] - H_RANGE[0])) * env._stairs.pyr  # flat tiles: no riser
  e = torch.relu(robot.data.joint_pos[:, ids] - (KNEE_FLEX + KNEE_FLEX_H * h_cm)[:, None])
  return _stairs_only((e * (env._stairs.load > 0.1 * BW)).sum(-1), env)


def stairs_feet_clear(env) -> torch.Tensor:
  """Swing feet below CLEAR_H over the ground under them, weighted by their horizontal speed (SOTA feet clearance)."""
  st = env._stairs
  robot = env.scene["robot"]
  fp = robot.data.body_link_pos_w[:, st.foot_ids]
  fv = robot.data.body_link_lin_vel_w[:, st.foot_ids, :2].norm(dim=-1)
  h = fp[..., 2] - 0.10 - env.scene["stairs_ground"].data.hit_pos_w[:, :2, 2]
  _, _, _, _, fn = _feet(env)
  swing = fn.view(fp.shape[0], 2, -1).sum(-1) <= 10.0
  e = (torch.relu(CLEAR_H - h) / CLEAR_H) ** 2
  return (e * fv * swing).sum(-1) * (_land_ramp(env) if FCLEAR_RAMP else 1.0)


def stairs_tread(env) -> torch.Tensor:
  """One-shot per tread: a loaded foot on a tread nearer the goal than any it stood on this episode
  (cf. Humanoid Parkour Learning's per-stair footstep reward). rl12 never put a foot on the 2.4 cm step:
  pelvis progress pays only after the step is taken."""
  st = env._stairs
  robot = env.scene["robot"]
  span = PLATFORM / 2 + (st.nsteps - 1) * st.tread
  sf = ((robot.data.body_link_pos_w[:, st.foot_ids, :2] - st.center[:, None]) * st.u[:, None]).sum(-1) - span[:, None]
  lvl = (-sf / st.tread.clamp(min=1e-3)[:, None]).ceil().clamp(0, NSTEPS)
  lvl = torch.where(st.climb[:, None], lvl, NSTEPS - lvl)  # levels gained toward the goal
  _, _, _, _, fn = _feet(env)
  loaded = fn.view(sf.shape[0], 2, -1).sum(-1) > 50.0
  best = getattr(env, "_tread_best", None)
  fresh = (env.episode_length_buf <= 1)[:, None]
  if best is None or best.shape != lvl.shape:
    best = lvl.clone()
  best = torch.where(fresh, lvl, best)
  new = torch.where(loaded & st.pyr[:, None], torch.maximum(best, lvl), best)
  gain = (new - best).sum(-1)
  env._tread_best = new
  return gain / env.step_dt


def stairs_air(env) -> torch.Tensor:
  """One foot in a 0.05-0.6 s swing while asked to go (cf. feet air time in the SOTA):
  from scratch the gait otherwise slides its feet."""
  st = env._stairs
  one = (st.air > 0).sum(-1) == 1
  r = _land_ramp(env) if AIR_WIN0 != AIR_WIN else 1.0
  lo, hi = (w0 + (w1 - w0) * r for w0, w1 in zip(AIR_WIN0, AIR_WIN))
  ok = ((st.air > lo) & (st.air < hi)).any(-1) & one
  on = ~st.arrived & ~st.paused & (env.episode_length_buf >= st.cmd_on)
  return _stairs_only((ok & on).float(), env)


def stairs_stop(env, std: float) -> torch.Tensor:
  st = env._stairs
  still = st.arrived
  if PAUSE_P > 0 or STAIR_PAUSE_P > 0:
    still = st.arrived | st.paused | ((env.episode_length_buf < st.cmd_on) & (env.episode_length_buf >= st.hold_end))
  r = still * torch.exp(-(st.v_ema**2).sum(-1) / std**2)
  if STOP_DS:
    # rl29 det balanced on one leg after arrival: still on two feet costs ~0 (zero-action test), pay that.
    # Graded on the lighter foot's load (full at 25% BW): rl30's binary gate was never reached.
    r = r * (st.load.min(-1).values / (0.25 * BW)).clamp(0.0, 1.0)
  return _stairs_only(r, env)


def _still(env) -> torch.Tensor:
  st = env._stairs
  still = st.arrived
  if PAUSE_P > 0 or STAIR_PAUSE_P > 0:
    still = st.arrived | st.paused | ((env.episode_length_buf < st.cmd_on) & (env.episode_length_buf >= st.hold_end))
  return still


def stairs_stop_pose(env) -> torch.Tensor:
  """Stopped like slot 18 (tardis 2026-11-19): feet side by side (stagger along the heading, std 3 cm) at the nominal
  width (std 3 cm), CoM over the middle of the feet (std 2 cm), both feet loaded. Paid only while the command is off.
  s50e stopped with the feet 6 cm staggered and rocked fore-aft instead of taking a closing step."""
  st = env._stairs
  robot = env.scene["robot"]
  fp = robot.data.body_link_pos_w[:, st.foot_ids, :2]
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  hf = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
  d = fp[:, 0] - fp[:, 1]
  dx = (d * hf).sum(-1)
  w = (-d[:, 0] * hf[:, 1] + d[:, 1] * hf[:, 0]).abs()
  com = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id, :2]
  c = (com - fp.mean(1)).norm(dim=-1)
  k = torch.exp(-((dx / 0.03) ** 2) - ((w - STANCE_W0) / 0.03) ** 2 - (c / 0.02) ** 2)
  both = (st.load.min(-1).values / (0.25 * BW)).clamp(0.0, 1.0)
  return _stairs_only(_still(env) * k * both, env) * _land_ramp(env)  # ramped (tardis: full guards collapsed the copy)


def stairs_still_slip(env) -> torch.Tensor:
  """Horizontal speed of each foot carrying > 25 % BW while the command is off, m/s summed (weight < 0): a closing step
  must cost less than sliding a loaded foot back into place (s50e slid 13 cm after a release)."""
  st = env._stairs
  robot = env.scene["robot"]
  v = robot.data.body_link_lin_vel_w[:, st.foot_ids, :2].norm(dim=-1)
  loaded = st.load > 0.25 * BW
  return _stairs_only(_still(env) * (v * loaded).sum(-1), env) * _land_ramp(env)


def stairs_otrack(env) -> torch.Tensor:
  """OSEQ envs while moving: body-frame velocity tracking, exp(-sum((v_i - target_i) / scale_i)^2 / OSEQ_STD^2), smoothed
  over OSEQ_TAU. Targets: sign x (fwd or back, side, yaw) speed; scale = side speed for vx, vy and yaw speed for wz.
  t3 (positional goals) learned not to fall by not moving: a speed target pays the motion itself."""
  st = env._stairs
  on = ~st.paused & (env.episode_length_buf >= st.cmd_on) & st.oseq & (st.ocmd != 0).any(-1)
  c = st.ocmd
  vx = torch.where(c[:, 0] > 0, OSEQ_V[0], OSEQ_V[1]) * c[:, 0]
  tgt = torch.stack([vx, OSEQ_V[2] * c[:, 1], OSEQ_V[3] * c[:, 2]], 1)
  sc = torch.tensor([OSEQ_V[2], OSEQ_V[2], OSEQ_V[3]], device=c.device)
  e = (((st.ov - tgt) / sc) ** 2).sum(-1)
  return torch.exp(-e / OSEQ_STD**2) * on


def stairs_same_tread(env) -> torch.Tensor:
  """Stopped with both feet loaded on one tread (Leo: preferred, one foot per tread allowed)."""
  st = env._stairs
  robot = env.scene["robot"]
  span = PLATFORM / 2 + (st.nsteps - 1) * st.tread
  sf = ((robot.data.body_link_pos_w[:, st.foot_ids, :2] - st.center[:, None]) * st.u[:, None]).sum(-1) - span[:, None]
  lvl = (-sf / st.tread.clamp(min=1e-3)[:, None]).ceil().clamp(0, NSTEPS)
  both = (st.load > 0.25 * BW).all(-1)
  return _stairs_only((st.paused & both & (lvl[:, 0] == lvl[:, 1])).float(), env)


def stairs_stall(env) -> torch.Tensor:
  """No centimetre gained for STALL_S while asked to go: pauses are free, standing is not. STALL_DESC_X scales it on
  descents: rl46 froze on the top platform, 8 s of stall (discounted ~-130) being cheaper than a fall (-200)."""
  st = env._stairs
  idle = (env.episode_length_buf - st.last_gain).float() * env.step_dt > STALL_S
  x = torch.where(st.climb, 1.0, STALL_DESC_X) if STALL_DESC_X != 1.0 else 1.0
  return _stairs_only(idle.float() * x, env)


def stairs_overspeed(env) -> torch.Tensor:
  st = env._stairs
  return _stairs_only(((st.v_ema.norm(dim=-1) - VMAX).clamp(min=0.0) / VMAX) ** 2, env)


def stairs_fall_cost(env) -> torch.Tensor:
  """One-shot on a fall (weight = the cost). FALL_S0 < 1: cheap falls first, so a scratch run
  lunges forward before it must walk (rl16 at full cost froze)."""
  s = FALL_S0 + (1.0 - FALL_S0) * _land_ramp(env) if FALL_S0 < 1 else 1.0
  return env.termination_manager.terminated.float() / env.step_dt * s


def stairs_heading(env, std: float) -> torch.Tensor:
  st = env._stairs
  q = env.scene["robot"].data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  e = torch.atan2(torch.sin(yaw - torch.atan2(st.face[:, 1], st.face[:, 0])),
                  torch.cos(yaw - torch.atan2(st.face[:, 1], st.face[:, 0])))
  return _stairs_only(torch.exp(-(e**2) / std**2), env)


def stairs_heading_err(env) -> torch.Tensor:
  """Heading error to the stairs axis beyond HEAD_DB, rad, linear (weight < 0). ap5's stronger heading reward (W 2, std 0.7)
  paid standing aligned at 2/s and froze the run: a cost has no value for standing still."""
  st = env._stairs
  q = env.scene["robot"].data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  d = yaw - torch.atan2(st.face[:, 1], st.face[:, 0])
  e = torch.atan2(torch.sin(d), torch.cos(d)).abs()
  return _stairs_only(torch.relu(e - HEAD_DB), env)


def stairs_alive(env) -> torch.Tensor:
  return env._stairs.is_stairs.float()


def stairs_arrival(env) -> torch.Tensor:
  """One-shot: worth its weight once per episode (reward = value * weight * dt)."""
  if SETTLE_T > 0:
    # Paid once still on two feet after the goal: rl21-32 crossed it and balanced on one leg.
    return env._stairs.settled_now.float() / env.step_dt
  return env._stairs.arrived_now.float() / env.step_dt


def _feet(env):
  s = env.scene["stairs_feet"].data
  n = s.normal
  f = s.force
  found = s.found > 0
  fn = (f * n).sum(-1).abs() * found
  return s.pos, n, f, found, fn


# Walking envelope of slot 18 (the best real-robot copy, 2026-10-29_06-55-45 model_12499) measured in this env
# on flat tiles (training randomization, its KI 2 ankle plant): free inside, squared beyond, ramped over BWC_RAMP
# env steps (Leo, 2026-11-11: guide the learning into a minimal admissible walking mode, freeze nothing).
# Knee and hip positions stay free: stairs need more than flat walking. rl41 on 2-4 cm stairs: ankle pitch speed
# p99 3.66 vs 1.41, hip pitch 2.1 vs 1.0, ankle pitch -82..-12 vs -36..-13 deg, swing sole 39 / 18 vs 12 / 5.5 deg.
BWC_JV = {"KNEE_P": 1.76, "ANKLE_P": 1.41, "CROTCH_P": 1.00, "CROTCH_R": 0.71, "ANKLE_R": 1.10, "CROTCH_Y": 0.70}
BWC_JV_MARGIN = float(_env("BWC_JV_MARGIN", "1.5"))
BWC_ANKLE_P = (math.radians(-36.2 - float(_env("BWC_ANKLE_P_MARGIN", "12"))),
               math.radians(-12.7 + float(_env("BWC_ANKLE_P_MARGIN", "12"))))
BWC_ANKLE_R = math.radians(12.4 + 5.0)
BWC_SWING_PITCH = math.radians(12.3 + 10.0)  # swing sole vs horizontal
BWC_SWING_ROLL = math.radians(5.5 + 4.0)
BWC_COP_X = (-0.009 - 0.02, 0.065 + 0.02)  # m, loaded foot's CoP in the ankle frame (BWC LIPM ZMP); slot 18 itself
BWC_COP_Y = 0.048 + 0.015  # reaches the sole edges at heel strike / toe-off, so this one does not discriminate
BWC_RAMP = float(_env("BWC_RAMP", "24000"))


def _bwc_ramp(env, key: str) -> float:
  c0 = getattr(env, key, None)
  if c0 is None:
    c0 = env.common_step_counter
    setattr(env, key, c0)
  return min(1.0, (env.common_step_counter - c0) / BWC_RAMP) if BWC_RAMP > 0 else 1.0


def _leg_ids(env, groups):
  key = "_bwc_ids_" + "_".join(groups)
  if not hasattr(env, key):
    names = env.scene["robot"].joint_names
    ids = [i for i, n in enumerate(names) if any(n.endswith(g) for g in groups)]
    grp = [next(g for g in groups if names[i].endswith(g)) for i in ids]
    setattr(env, key, (torch.tensor(ids, device=env.device), grp))
  return getattr(env, key)


def stairs_bwc_jvel(env) -> torch.Tensor:
  """Leg joint speed beyond BWC_JV_MARGIN x slot 18's p99, per joint, in units of that p99."""
  ids, grp = _leg_ids(env, tuple(BWC_JV))
  cap = getattr(env, "_bwc_jv_cap", None)
  if cap is None:
    cap = env._bwc_jv_cap = torch.tensor([BWC_JV[g] for g in grp], device=env.device)
  qd = env.scene["robot"].data.joint_vel[:, ids].abs()
  e = (torch.relu(qd - BWC_JV_MARGIN * cap) / cap) ** 2
  return _stairs_only(e.sum(-1), env) * _bwc_ramp(env, "_bwc_c0_jv")


def stairs_bwc_ankle(env) -> torch.Tensor:
  """Ankle pitch outside slot 18's range + margin, roll beyond its p99 + 5 deg, per 10 / 5 deg."""
  ip, _ = _leg_ids(env, ("ANKLE_P",))
  ir, _ = _leg_ids(env, ("ANKLE_R",))
  q = env.scene["robot"].data.joint_pos
  p = q[:, ip]
  ep = (torch.relu(BWC_ANKLE_P[0] - p) + torch.relu(p - BWC_ANKLE_P[1])) / math.radians(10.0)
  er = torch.relu(q[:, ir].abs() - BWC_ANKLE_R) / math.radians(5.0)
  return _stairs_only((ep**2).sum(-1) + (er**2).sum(-1), env) * _bwc_ramp(env, "_bwc_c0_ank")


def _foot_tilt(env):
  q = env.scene["robot"].data.body_link_quat_w[:, env._stairs.foot_ids]  # wxyz
  w, x, y, z = q.unbind(-1)
  return torch.asin((2 * (x * z - w * y)).clamp(-1, 1)), torch.asin((2 * (y * z + w * x)).clamp(-1, 1))


def stairs_bwc_swing_foot(env) -> torch.Tensor:
  """Swing sole tilted beyond slot 18's (pitch 22, roll 9.5 deg), per 10 / 5 deg: no toes-up riser dodging."""
  st = env._stairs
  pit, rol = _foot_tilt(env)
  sw = (st.load < UP_LOAD).float() if hasattr(st, "load") and st.load is not None else 1.0
  e = (torch.relu(pit.abs() - BWC_SWING_PITCH) / math.radians(10.0)) ** 2 + (
    torch.relu(rol.abs() - BWC_SWING_ROLL) / math.radians(5.0)) ** 2
  return _stairs_only((e * sw).sum(-1), env) * _bwc_ramp(env, "_bwc_c0_sw")


def stairs_bwc_cop(env) -> torch.Tensor:
  """Loaded foot's centre of pressure outside the BWC's ZMP zone (+2 cm fore-aft, +1.5 cm lateral), in the
  ankle frame, per 2 / 1.5 cm."""
  pos, _, _, _, fn = _feet(env)
  b = fn.shape[0]
  fn = fn.view(b, 2, -1)
  pos = pos.view(b, 2, -1, 3)
  load = fn.sum(-1)
  cop = (pos * fn[..., None]).sum(-2) / load.clamp(min=1e-6)[..., None]
  robot = env.scene["robot"]
  idx = env._stairs.foot_ids
  a = robot.data.body_link_pos_w[:, idx]
  q = robot.data.body_link_quat_w[:, idx]
  w, x, y, z = q.unbind(-1)
  fx = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)
  fy = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), -1)
  d = cop - a
  rx, ry = (d * fx).sum(-1), (d * fy).sum(-1)
  e = ((torch.relu(BWC_COP_X[0] - rx) + torch.relu(rx - BWC_COP_X[1])) / 0.02) ** 2 + (
    torch.relu(ry.abs() - BWC_COP_Y) / 0.015) ** 2
  on = (load > 0.15 * BW).float()
  return _stairs_only((e * on).sum(-1), env) * _bwc_ramp(env, "_bwc_c0_cop")


def stairs_riser_kick(env) -> torch.Tensor:
  """Force on the feet from near-vertical faces (risers, nosing edges), in body weights."""
  _, n, f, found, _ = _feet(env)
  side = (n[..., 2].abs() < 0.5) & found
  r = (f.norm(dim=-1) * side).sum(-1) / BW
  return _stairs_only(r, env)


def stairs_leg_contact(env) -> torch.Tensor:
  d = env.scene["stairs_legs"].data
  r = (d.force.norm(dim=-1) > 1.0).float().sum(-1)
  return _stairs_only(r, env)


def stairs_foot_tip(env) -> torch.Tensor:
  """Stance foot rotating about a horizontal axis: tipping over an edge or rocking heel/toe."""
  _, n, _, found, fn = _feet(env)
  b = fn.shape[0]
  load = fn.view(b, 2, -1).sum(-1)
  robot = env.scene["robot"]
  idx = env._stairs.foot_ids
  w = robot.data.body_link_ang_vel_w[:, idx, :2].norm(dim=-1)
  return _stairs_only((w * (load > 50.0)).sum(-1), env)


def _hull_margin(pts: torch.Tensor, valid: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Signed distance-like margin of x [B,2] to the convex hull of the valid pts [B,K,2].

  A pair (i, j) is a CCW hull edge when every valid point lies left of it; the margin
  is the minimum over hull edges of the signed distance to the edge line (> 0 inside).
  """
  d = pts[:, None, :, :] - pts[:, :, None, :]  # [B,i,j,2] = pj - pi
  length = d.norm(dim=-1)
  rel = pts[:, None, :, :] - pts[:, :, None, :]  # pk - pi, indexed [B,i,k]
  cross = d[:, :, :, None, 0] * rel[:, :, None, :, 1] - d[:, :, :, None, 1] * rel[:, :, None, :, 0]
  ok_k = (cross >= -1e-5) | ~valid[:, None, None, :]
  edge = ok_k.all(-1) & valid[:, :, None] & valid[:, None, :] & (length > 2e-3)
  xr = x[:, None, None, :] - pts[:, :, None, :]
  sd = (d[..., 0] * xr[..., 1] - d[..., 1] * xr[..., 0]) / length.clamp(min=1e-6)
  sd = torch.where(edge, sd, torch.full_like(sd, 1e3))
  m = sd.flatten(1).min(-1).values
  has = edge.flatten(1).any(-1)
  return torch.where(has, m, torch.full_like(m, -0.05)), has


SOLE_RAY_DEPTH = 0.097 + 0.012  # ankle to sole bottom, plus 1.2 cm of allowed gap


def _support(env):
  """Points of the soles actually resting on something: 6 rays per foot from the
  ankle along the foot's -z, kept when the foot is loaded and the ray lands within
  1.2 cm of the sole. Over a nosing the rays past the edge fall to the step below.
  Returns the points [B,12,3], their validity and the per-foot load flag [B,2]."""
  d = env.scene["stairs_soles"].data
  b = d.distances.shape[0]
  _, _, _, _, fn = _feet(env)
  loaded = fn.view(b, 2, -1).sum(-1) > 10.0
  hit = (d.distances >= 0) & (d.distances < SOLE_RAY_DEPTH)
  valid = hit & loaded.repeat_interleave(d.distances.shape[1] // 2, dim=1)
  return d.hit_pos_w, valid, loaded


def stairs_swing_clearance(env) -> torch.Tensor:
  """Swing foot over what lies under and just ahead of it (toe + 10 cm), when that is
  higher than the ground under the ankle: +1 at 3 cm over the top, 0 level, -1 below.
  Gives the lift a gradient where the blind gait would stall or stub."""
  st = env._stairs
  robot = env.scene["robot"]
  b = robot.data.root_link_pos_w.shape[0]
  fp = robot.data.body_link_pos_w[:, st.foot_ids]
  scan = env.scene["stairs_scan"].data.hit_pos_w
  q = robot.data.root_link_quat_w
  yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  c, sn = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
  rx = scan[:, None, :, 0] - fp[:, :, None, 0]
  ry = scan[:, None, :, 1] - fp[:, :, None, 1]
  lx, ly = c * rx + sn * ry, -sn * rx + c * ry
  win = (lx > -0.10) & (lx < 0.23) & (ly.abs() < 0.07)
  top = torch.where(win, scan[:, None, :, 2], torch.full_like(lx, -1e3)).max(-1).values
  ground = env.scene["stairs_ground"].data.hit_pos_w[:, :2, 2]
  ahead = win.any(-1) & (top > ground + 0.01)
  _, _, _, _, fn = _feet(env)
  loaded = fn.view(b, 2, -1).sum(-1) > 10.0
  on = ~loaded & ahead & (env.episode_length_buf >= st.cmd_on)[:, None] & ~st.arrived[:, None] & ~st.paused[:, None]
  if CLEAR_FWD:
    # rl12 stepped in place at the riser, feet high, for +3.3/s: only a foot moving toward the goal counts.
    fv = (robot.data.body_link_lin_vel_w[:, st.foot_ids, :2] * st.face[:, None]).sum(-1)
    on = on & (fv > 0.05)
  r = ((fp[..., 2] - 0.10 - top) / 0.03).clamp(-1.0, 1.0)
  return _stairs_only((r * on).sum(-1), env)


def stairs_double_support(env) -> torch.Tensor:
  """Both feet loaded: the copy's double_support needs the reference and is masked here,
  and slip_speed / no_slip alone push the gait to 86 % single support."""
  st = env._stairs
  b = env.scene["robot"].data.root_link_pos_w.shape[0]
  _, _, _, _, fn = _feet(env)
  both = (fn.view(b, 2, -1).sum(-1) > 10.0).all(-1)
  return _stairs_only((both & (env.episode_length_buf >= st.hold_end)).float(), env)


def stairs_foothold(env) -> torch.Tensor:
  """Loaded foot with more than half its sole over the void (cf. arXiv 2603.07928):
  0 up to one ray row out (heel or toe overhang is fine), 1 from 4 of 6 rays out."""
  st = env._stairs
  d = env.scene["stairs_soles"].data
  b = d.distances.shape[0]
  _, _, _, _, fn = _feet(env)
  loaded = fn.view(b, 2, -1).sum(-1) > 10.0
  if FOOTHOLD_PRE:
    # Also the swing foot over an edge in its last 5 cm: the penalty comes before the bad landing.
    # Hit heights, not ray lengths: a pitched foot at heel strike lengthens the toe rays.
    z = torch.where(d.distances >= 0, d.hit_pos_w[..., 2], torch.full_like(d.distances, -1e3)).view(b, 2, -1)
    top = z.max(-1).values
    drop = 0.012
    if FOOTHOLD_REL:
      # Rays read the edge only below 1.2 cm: on 0.5-1.2 cm risers the foothold term never fired.
      lv = env.scene.terrain.terrain_levels.float()
      h = H_RANGE[0] + lv / max(ROWS - 1, 1) * (H_RANGE[1] - H_RANGE[0])
      drop = (0.5 * h).clamp(0.002, 0.012)[:, None, None]
    out = (z < top[..., None] - drop).sum(-1).float()
    ankle = env.scene["robot"].data.body_link_pos_w[:, st.foot_ids, 2]
    clear = ankle - (SOLE_RAY_DEPTH - 0.012) - top
    wgt = torch.where(loaded, 1.0, (1.0 - clear / 0.05).clamp(0.0, 1.0))
  else:
    hit = (d.distances >= 0) & (d.distances < SOLE_RAY_DEPTH)
    out = (~hit).view(b, 2, -1).sum(-1).float()
    wgt = loaded.float()
  free = 2.0
  if FOOTHOLD_DESC_FREE != 2.0:
    free = torch.where(st.climb, 2.0, FOOTHOLD_DESC_FREE)[:, None]
  r = ((out - free) / 2.0).clamp(0.0, 1.0) * wgt
  return _stairs_only(r.sum(-1) * (env.episode_length_buf >= st.hold_end), env)


def _com(env):
  robot = env.scene["robot"]
  return robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]


def stairs_capture_point(env) -> torch.Tensor:
  """Dynamic balance (required): in double support, and while standing, the capture
  point stays inside the hull of the supported sole points. In single support the CP
  normally runs ahead of the stance foot (the next step catches it), so it is not
  judged there. 0..1, saturating 10 cm outside."""
  st = env._stairs
  pos, valid, loaded = _support(env)
  com = _com(env)
  zc = (pos[..., 2] * valid).sum(-1) / valid.sum(-1).clamp(min=1)
  h = (com[:, 2] - zc).clamp(min=0.3)
  cp = com[:, :2] + st.com_vel[:, :2] / torch.sqrt(9.81 / h)[:, None]
  m, _ = _hull_margin(pos[..., :2], valid, cp)
  standing = st.arrived | st.paused | (env.episode_length_buf < st.cmd_on)
  # Not during the arming hold: the held plant rocks foot to foot and the policy cannot act.
  armed = env.episode_length_buf >= st.hold_end
  active = (loaded.all(-1) | (standing & loaded.any(-1))) & valid.any(-1) & armed
  r = (-m).clamp(0.0, 0.1) / 0.1 * active
  return _stairs_only(r, env)


def stairs_static_balance(env) -> torch.Tensor:
  """Static balance (bonus): in single support, the CoM over the supported part of
  the stance sole -- the robot could stop dead on that foot."""
  pos, valid, loaded = _support(env)
  m, _ = _hull_margin(pos[..., :2], valid, _com(env)[:, :2])
  single = (loaded.sum(-1) == 1) & (env.episode_length_buf >= env._stairs.hold_end)
  if STATIC_AIR > 0:
    # Paid during a step only: standing on one leg at the riser collected it for free.
    single = single & (env._stairs.up.max(-1).values < STATIC_AIR)
  return _stairs_only(((m > 0.01) & single).float(), env)


_TEACHER = {}


def _teacher(env):
  """Frozen start policy (its scan columns are zero, so it is the blind walk)."""
  if "net" not in _TEACHER:
    sd = torch.load(TEACHER, map_location=env.device, weights_only=False)["actor_state_dict"]
    lin = [k[:-7] for k in sd if k.startswith("mlp.") and k.endswith(".weight")]
    _TEACHER["net"] = (sd["obs_normalizer._mean"], sd["obs_normalizer._std"],
                       [(sd[k + ".weight"], sd[k + ".bias"]) for k in lin])
  return _TEACHER["net"]


def stairs_teacher_match(env) -> torch.Tensor:
  """Penalty (weight < 0): squared gap to the start policy's action where the scan
  sees flat ground, so the stairs reshape the gait near steps only."""
  mean, std, layers = _teacher(env)
  x = (env.obs_buf["actor"] - mean) / (std + 1e-2)
  with torch.no_grad():
    for i, (w, b) in enumerate(layers):
      x = x @ w.T + b
      if i < len(layers) - 1:
        x = torch.nn.functional.elu(x)
  armed = env.episode_length_buf >= env._stairs.hold_end
  if TEACHER_MASK == "flat":
    z = env.scene["stairs_scan"].data.hit_pos_w[..., 2]
    armed = armed & ((z.max(-1).values - z.min(-1).values) < 0.01)
  gap = (env.action_manager.action - x).pow(2).mean(-1)
  return _stairs_only(gap * armed, env)


# ---------------------------------------------------------------------------
# Terminations, curriculum, observations
# ---------------------------------------------------------------------------


def stairs_fall(env) -> torch.Tensor:
  robot = env.scene["robot"]
  ground = env.scene["stairs_ground"].data.hit_pos_w[:, 2, 2]
  low = robot.data.root_link_pos_w[:, 2] - ground < 0.5
  tilt = robot.data.projected_gravity_b[:, 2] > -0.55
  body = env.scene["stairs_body"].data.found.sum(-1) > 0
  return (env._stairs.is_stairs | RL) & (low | tilt | body)


def stairs_one_leg(env) -> torch.Tensor:
  """Constraint as termination (cf. CaT, arXiv 2403.18765): rl25-27 kept standing on one leg through the
  long-swing cost."""
  st = env._stairs
  lim = torch.full_like(st.settle_t, ONE_LEG_T)
  if ONE_LEG_OFF > 0:
    lim = torch.where(st.arrived | st.paused, torch.full_like(lim, ONE_LEG_OFF), lim)
  return st.is_stairs & (st.up.max(-1).values > lim) & (env.episode_length_buf >= st.hold_end)


def stairs_saturation(env) -> torch.Tensor:
  """Constraint as termination: a leg joint held at its torque limit longer than the (ramped) threshold."""
  robot = env.scene["robot"]
  if not hasattr(env, "_st_sat_lim"):
    env._st_sat_lim = torch.tensor([next((v for g, v in _TRQ_LIM.items() if g in n), float("inf"))
                                    for n in robot.joint_names], device=env.device)
    env._st_sat_run = torch.zeros(env.num_envs, device=env.device)
    env._st_sat_c0 = env.common_step_counter
  sat = (robot.data.qfrc_actuator.abs() >= 0.97 * env._st_sat_lim).any(-1)
  fresh = env.episode_length_buf <= 1
  env._st_sat_run = torch.where(sat & ~fresh, env._st_sat_run + env.step_dt, torch.zeros_like(env._st_sat_run))
  t = SAT_T
  if SAT_T0 > SAT_T and SAT_RAMP > 0:
    r = min(1.0, (env.common_step_counter - env._st_sat_c0) / SAT_RAMP)
    t = SAT_T0 + (SAT_T - SAT_T0) * r
  return env._stairs.is_stairs & (env._st_sat_run > t) & (env.episode_length_buf >= env._stairs.hold_end)


def stairs_time_out(env) -> torch.Tensor:
  lim = torch.where(env._stairs.is_stairs, math.ceil(EPISODE_S / env.step_dt), env.max_episode_length)
  return env.episode_length_buf >= lim


def stairs_levels(env, env_ids: torch.Tensor) -> dict:
  st = env._stairs
  terrain = env.scene.terrain
  s = st.pyr[env_ids] & ~st.standing[env_ids]
  # Promote on an arrival the robot survives (descents used to "arrive" and fall off the last step).
  held = st.arrived[env_ids] & ~env.termination_manager.terminated[env_ids]
  up = (held if HELD_ARRIVAL else st.arrived[env_ids]) & s
  down = ~st.arrived[env_ids] & (st.progress[env_ids] < 0.5) & s
  if env.common_step_counter == 0 or LVL_FREEZE:
    up = torch.zeros_like(up)
    down = torch.zeros_like(down)
  if REPLAY_P > 0:
    if st.lvl_cur is None:
      st.lvl_cur = terrain.terrain_levels.clone()
      if LVL_FILE and os.path.exists(LVL_FILE):
        # A resumed run starts where the last one stopped: rl41-46 re-climbed from INIT_LEVEL for ~700 it each.
        d = torch.load(LVL_FILE, map_location=env.device)
        if d["lvl"].shape == st.lvl_cur.shape:
          st.lvl_cur = d["lvl"].to(st.lvl_cur.dtype)
          st.mode_climb = d["climb"].clone()
    st.n_lvl_calls = getattr(st, "n_lvl_calls", 0) + 1
    if LVL_FILE and st.n_lvl_calls % 2000 == 0:
      torch.save({"lvl": st.lvl_cur, "climb": st.mode_climb}, LVL_FILE + ".tmp")
      os.replace(LVL_FILE + ".tmp", LVL_FILE)
    keep = st.replayed[env_ids]  # a replay episode moves nothing
    if PROMOTE_N > 1:
      if getattr(st, "streak", None) is None:
        st.streak = torch.zeros_like(st.lvl_cur)
      k = st.streak[env_ids]
      k = torch.where(keep, k, torch.where(up, k + 1, torch.zeros_like(k)))
      up = up & (k >= PROMOTE_N)
      st.streak[env_ids] = torch.where(up, torch.zeros_like(k), k)
    cur = st.lvl_cur[env_ids] + (up & ~keep).long() - (down & ~keep).long()
    top = terrain.max_terrain_level if LVL_MAX < 0 else min(terrain.max_terrain_level, LVL_MAX + 1)
    cur = torch.where(cur >= top, torch.randint_like(cur, top), cur.clamp(min=0))
    st.lvl_cur[env_ids] = cur
    rep = (torch.rand(len(env_ids), device=env.device) < REPLAY_P) & s
    lvl = torch.where(rep, (torch.rand(len(env_ids), device=env.device) * (cur + 1).float()).long(), cur)
    terrain.terrain_levels[env_ids] = lvl.to(terrain.terrain_levels.dtype)
    terrain.env_origins[env_ids] = terrain.terrain_origins[lvl, terrain.terrain_types[env_ids]]
    st.replayed[env_ids] = rep
  else:
    terrain.update_env_origins(env_ids, up, down)
  lv = (st.lvl_cur if st.lvl_cur is not None else terrain.terrain_levels).float()
  out = {}
  m = st.kind == 1
  if m.any():
    out["level"] = lv[m].mean()
    if MODE_FIXED:
      out["level_climb"] = lv[m & st.mode_climb].mean()
      out["level_descend"] = lv[m & ~st.mode_climb].mean()
    out["step_height"] = (H_RANGE[0] + lv[m] / max(ROWS - 1, 1) * (H_RANGE[1] - H_RANGE[0])).mean()
  if s.any():
    out["arrival"] = st.arrived[env_ids][s].float().mean()
    out["progress"] = st.progress[env_ids][s].mean()
    out["arrival_held"] = held[s].float().mean()
    climb = st.climb[env_ids] & s
    if climb.any():
      out["arrival_climb"] = st.arrived[env_ids][climb].float().mean()
    desc = ~st.climb[env_ids] & s
    if desc.any():
      out["arrival_descend"] = st.arrived[env_ids][desc].float().mean()
  return out


def stairs_state(env) -> torch.Tensor:
  st = env._stairs
  on = (~st.arrived & ~st.paused & (env.episode_length_buf >= st.cmd_on)).float()
  return torch.stack([st.is_stairs.float(), st.arrived.float(), on], -1)


def scan_dim() -> int:
  nx = len(torch.arange(-SCAN_SIZE[0] / 2, SCAN_SIZE[0] / 2 + SCAN_RES * 0.5, SCAN_RES))
  ny = len(torch.arange(-SCAN_SIZE[1] / 2, SCAN_SIZE[1] / 2 + SCAN_RES * 0.5, SCAN_RES))
  return nx * ny


@dataclass
class _ShiftedGridCfg(GridPatternCfg):
  """Grid moved SCAN_FWD ahead: a pelvis-centred map spent half its rays behind the robot."""

  fwd: float = 0.0

  def generate_rays(self, mj_model, device):
    o, d = super().generate_rays(mj_model, device)
    o = o.clone()
    o[:, 0] += self.fwd
    return o, d


def scan_mirror_perm() -> list[int]:
  nx = len(torch.arange(-SCAN_SIZE[0] / 2, SCAN_SIZE[0] / 2 + SCAN_RES * 0.5, SCAN_RES))
  ny = len(torch.arange(-SCAN_SIZE[1] / 2, SCAN_SIZE[1] / 2 + SCAN_RES * 0.5, SCAN_RES))
  return [(ny - 1 - iy) * nx + ix for iy in range(ny) for ix in range(nx)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def apply(cfg, play: bool = False) -> None:
  from dataclasses import fields as _fields

  # No rl_ext import here: it imports the velocity task, whose ablation rewrites the
  # shared RHPS1_ACTION_SCALE (knee 0.0525 -> 0.081) for every cfg built after it.
  from . import env_cfgs as E

  # Terrain: one column per tread, then the flat column (the copy). Stairs first so
  # env 0, the one the training videos film, is on stairs; any flat tile adds the plane.
  kinds, treads, subs = [], [], {}
  for t in TREADS:
    subs[f"up{int(round(t * 100))}"] = StepPyramidCfg(
      proportion=FRAC / len(TREADS), step_height_range=H_RANGE, step_width=t,
      num_steps=NSTEPS, platform_width=PLATFORM,
    )
    kinds.append("up")
    treads.append(t)
  if _env("FLATBOX", "0") == "1":
    from mjlab.terrains import BoxFlatTerrainCfg

    subs["flat"] = BoxFlatTerrainCfg(proportion=max(1.0 - FRAC, 1e-3))
  else:
    subs["flat"] = GroundPlaneCfg(proportion=max(1.0 - FRAC, 1e-3))
  kinds.append("flat")
  treads.append(0.0)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(TILE, TILE), num_rows=ROWS, curriculum=True, border_width=0.0,
      sub_terrains=subs, color_scheme="height",
    ),
    max_init_terrain_level=int(_env("INIT_LEVEL", "2")),
  )
  # Foot-on-step is box-box: the default convex path gives 1-8 drifting contacts per
  # flat foot and the copy fell on flat box tiles where it stands on the plane.
  if _env("NATIVE_BOXBOX", "1") == "1":
    cfg.sim.mujoco.disableflags = tuple(set(cfg.sim.mujoco.disableflags) | {"nativeccd"})
  # Raising the solver buffers costs 1.2 GB at 4096 envs (300/160: OOM on the 11 GB card).
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, int(_env("NCONMAX", "0")))
  cfg.sim.njmax = max(cfg.sim.njmax or 0, int(_env("NJMAX", "0")))

  # The copy already trains the knees at 108 N.m (Leo OK'd 100 for stairs); forcing 70
  # or 100 on slot 20 made it fall on flat ground. Override only on request.
  if os.environ.get("RHPS1_KNEE_LIMIT"):
    for a in cfg.scene.entities["robot"].articulation.actuators:
      if any("KNEE" in e for e in a.target_names_expr):
        a.effort_limit = float(os.environ["RHPS1_KNEE_LIMIT"])

  stage = int(_env("STAGE", "9"))  # bisection: 1 terrain, 2 + sensors, 3 + command, 9 all
  if stage < 2:
    return
  feet = r"^(L_ANKLE_P_LINK|R_ANKLE_P_LINK)$"
  cfg.scene.sensors = cfg.scene.sensors + (
    RayCastSensorCfg(
      name="stairs_scan", frame=ObjRef(type="body", name="BODY", entity="robot"),
      pattern=_ShiftedGridCfg(size=SCAN_SIZE, resolution=SCAN_RES, fwd=SCAN_FWD), ray_alignment="yaw",
      max_distance=3.0, include_geom_groups=(0,),
    ),
    RayCastSensorCfg(
      name="stairs_ground",
      frame=tuple(ObjRef(type="body", name=b, entity="robot")
                  for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK", "BODY")),
      pattern=GridPatternCfg(size=(0.0, 0.0), resolution=1.0), ray_alignment="world",
      max_distance=4.0, include_geom_groups=(0,),
    ),
    RayCastSensorCfg(
      name="stairs_soles",
      frame=tuple(ObjRef(type="body", name=b, entity="robot") for b in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")),
      pattern=GridPatternCfg(size=(0.18, 0.09), resolution=0.09), ray_alignment="base",
      max_distance=0.4, include_geom_groups=(0,),
    ),
    ContactSensorCfg(
      name="stairs_feet",
      primary=ContactMatch(mode="body", pattern=feet, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force", "pos", "normal", "tangent"),
      reduce="maxforce", num_slots=8, global_frame=True,
      history_length=cfg.decimation if FPEAK_SUB else 0,
    ),
    ContactSensorCfg(
      name="stairs_legs",
      primary=ContactMatch(mode="body", pattern=r"^[LR]_(KNEE_P|CROTCH_P)_LINK$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce",
    ),
    ContactSensorCfg(
      name="stairs_body",
      primary=ContactMatch(mode="body", pattern=r"^(BODY|CHEST_Y_LINK|CHEST_P_LINK)$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",), reduce="netforce",
    ),
  )

  if stage < 3:
    return
  # Command: re-anchored copy + stair spawns.
  m = cfg.commands["motion"]
  kw = {f.name: getattr(m, f.name) for f in _fields(m)}
  kw.setdefault("standing_start_prob", 0.0)
  if RL and _env("GHOST", "0") != "1":
    kw["debug_vis"] = False  # the BWC clip still plays inside the command (state host only): hide it
  cfg.commands["motion"] = StairsMotionCommandCfg(**kw, stairs_kinds=kinds, stairs_treads=treads)

  if stage < 4:
    return
  # Forward-then-zero direction command on stair envs, everywhere it is read.
  if not getattr(E, "_stairs_patched", False):
    E._reference_velocity_command = _command_wrapper(E._reference_velocity_command)
    E._stairs_patched = True
  for g in cfg.observations.values():
    for t in g.terms.values():
      if getattr(t.func, "__name__", "") == "_reference_velocity_command":
        t.func = E._reference_velocity_command

  for name, r in cfg.rewards.items():
    if (name.startswith("motion_") or name in _FLAT_ONLY) and not isinstance(r.func, type):
      r.func = _flat_only(r.func)
  for name in _FLAT_ONLY_TERMINATIONS:
    if name in cfg.terminations:
      cfg.terminations[name].func = _flat_only(cfg.terminations[name].func)

  w = lambda k, d: float(_env("W_" + k, d))  # noqa: E731
  if RL:
    for name in list(cfg.rewards):
      if name not in _RL_KEEP and not name.endswith("_proximity"):
        del cfg.rewards[name]
    for name in _FLAT_ONLY_TERMINATIONS:
      cfg.terminations.pop(name, None)
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    legs = SceneEntityCfg("robot", joint_names=(r"^[LR]_(CROTCH|KNEE|ANKLE)_.*",))
    cfg.rewards.update({
      "stairs_progress": RewardTermCfg(func=stairs_progress, weight=w("PROG", "30")),
      "stairs_stop": RewardTermCfg(func=stairs_stop, weight=w("STOP", "2"), params={"std": VEL_STD}),
      "stairs_stall": RewardTermCfg(func=stairs_stall, weight=w("STALL", "-2")),
      "stairs_overspeed": RewardTermCfg(func=stairs_overspeed, weight=w("OVERSPEED", "-10")),
      "stairs_fall_cost": RewardTermCfg(func=stairs_fall_cost, weight=w("FALL", "-30")),
      "stairs_upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=w("UPRIGHT", "-2")),
      "stairs_bwc_jvel": RewardTermCfg(func=stairs_bwc_jvel, weight=w("BWC_JV", "0")),
      "stairs_bwc_ankle": RewardTermCfg(func=stairs_bwc_ankle, weight=w("BWC_ANKLE", "0")),
      "stairs_bwc_swing_foot": RewardTermCfg(func=stairs_bwc_swing_foot, weight=w("BWC_SWING", "0")),
      "stairs_bwc_cop": RewardTermCfg(func=stairs_bwc_cop, weight=w("BWC_COP", "0")),
      "stairs_joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=w("JVEL", "-0.005"),
                                        params={"asset_cfg": legs}),
    })
    if w("AIR", "0") != 0:
      cfg.rewards["stairs_air"] = RewardTermCfg(func=stairs_air, weight=w("AIR", "0"))
    if w("CAD", "0") != 0:
      cfg.rewards["stairs_cadence"] = RewardTermCfg(func=stairs_cadence, weight=w("CAD", "0"))
    if w("LSWING", "0") != 0 and SWING_MAX > 0:
      cfg.rewards["stairs_long_swing"] = RewardTermCfg(func=stairs_long_swing, weight=w("LSWING", "0"))
    if w("STANCE", "0") != 0:
      cfg.rewards["stairs_stance"] = RewardTermCfg(func=stairs_stance, weight=w("STANCE", "0"))
    if w("TD", "0") != 0:
      cfg.rewards["stairs_touchdown"] = RewardTermCfg(func=stairs_touchdown, weight=w("TD", "0"))
    if w("IMP", "0") != 0:
      cfg.rewards["stairs_impact"] = RewardTermCfg(func=stairs_impact, weight=w("IMP", "0"))
    for k, f in (("TRUNK", stairs_trunk), ("VZ", stairs_vz), ("TRQ", stairs_torque), ("FCLEAR", stairs_feet_clear),
                 ("TREAD", stairs_tread), ("VINST", stairs_vinst), ("KNEE", stairs_knee_flex),
                 ("SAMETREAD", stairs_same_tread), ("HEADERR", stairs_heading_err), ("ROLL", stairs_roll),
                 ("STOPPOSE", stairs_stop_pose), ("STILLSLIP", stairs_still_slip), ("OTRACK", stairs_otrack)):
      if w(k, "0") != 0:
        cfg.rewards["stairs_" + k.lower()] = RewardTermCfg(func=f, weight=w(k, "0"))
    if w("JACC", "0") != 0:
      cfg.rewards["stairs_joint_acc"] = RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=w("JACC", "0"),
                                                      params={"asset_cfg": legs})
  else:
    cfg.rewards["stairs_lin_vel"] = RewardTermCfg(func=stairs_lin_vel, weight=w("VEL", "4"), params={"std": VEL_STD})
  cfg.rewards.update({
    "stairs_heading": RewardTermCfg(func=stairs_heading, weight=w("HEADING", "1"), params={"std": HEADING_STD}),
    "stairs_alive": RewardTermCfg(func=stairs_alive, weight=w("ALIVE", "3")),
    "stairs_arrival": RewardTermCfg(func=stairs_arrival, weight=w("ARRIVAL", "20")),
    "stairs_riser_kick": RewardTermCfg(func=stairs_riser_kick, weight=w("KICK", "-10")),
    "stairs_leg_contact": RewardTermCfg(func=stairs_leg_contact, weight=w("LEG", "-2")),
    "stairs_foot_tip": RewardTermCfg(func=stairs_foot_tip, weight=w("TIP", "-1")),
    "stairs_capture_point": RewardTermCfg(func=stairs_capture_point, weight=w("CP", "-2")),
    "stairs_static_balance": RewardTermCfg(func=stairs_static_balance, weight=w("STATIC", "0.5")),
  })
  if w("CLEAR", "0") != 0:
    cfg.rewards["stairs_swing_clearance"] = RewardTermCfg(func=stairs_swing_clearance, weight=w("CLEAR", "0"))
  if w("FOOTHOLD", "0") != 0:
    cfg.rewards["stairs_foothold"] = RewardTermCfg(func=stairs_foothold, weight=w("FOOTHOLD", "0"))
  if w("DS", "0") != 0:
    cfg.rewards["stairs_double_support"] = RewardTermCfg(func=stairs_double_support, weight=w("DS", "0"))
  if w("TEACHER", "0") != 0:
    cfg.rewards["stairs_teacher_match"] = RewardTermCfg(func=stairs_teacher_match, weight=w("TEACHER", "0"))
  cfg.terminations["stairs_fall"] = TerminationTermCfg(func=stairs_fall)
  if RL and ONE_LEG_T > 0:
    cfg.terminations["stairs_one_leg"] = TerminationTermCfg(func=stairs_one_leg)
  if SAT_T > 0:
    cfg.terminations["stairs_saturation"] = TerminationTermCfg(func=stairs_saturation)
  if EPISODE_S > 0:
    cfg.terminations["time_out"].func = stairs_time_out
  if cfg.curriculum is None:
    cfg.curriculum = {}
  cfg.curriculum["stairs_levels"] = CurriculumTermCfg(func=stairs_levels)

  scan = dict(func=envs_mdp.height_scan, params={"sensor_name": "stairs_scan", "offset": SCAN_OFFSET},
              clip=(-1.0, 1.0))
  cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
    **scan, noise=None if play or SCAN_NOISE == 0 else Unoise(n_min=-SCAN_NOISE, n_max=SCAN_NOISE))
  from mjlab.tasks.tracking.config.rhps1 import vision_map

  if vision_map.VIS:
    # Student: the actor reads the onboard map, the critic keeps the exact scan. Head held pitched down.
    cfg.scene.sensors = cfg.scene.sensors + (vision_map.camera_sensor_cfg(),)
    cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(func=vision_map.vision_scan)
  if vision_map.HEAD_HOLD:
    rob = cfg.scene.entities["robot"]
    rob.init_state = dataclasses.replace(
      rob.init_state, joint_pos={**rob.init_state.joint_pos, "HEAD_P": vision_map.HEAD_P})
  from mjlab.tasks.tracking.config.rhps1 import depth_student

  if depth_student.DEPTH:
    depth_student.apply(cfg, play=play)
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(**scan)
  cfg.observations["critic"].terms["stairs_state"] = ObservationTermCfg(func=stairs_state)
  if RL:
    # Same widths so the walking checkpoint loads; the reference means nothing here.
    for k in ("command", "motion_anchor_pos_b", "motion_anchor_ori_b"):
      t = cfg.observations["critic"].terms.get(k)
      if t is not None:
        t.func = _zeroed(t.func)

