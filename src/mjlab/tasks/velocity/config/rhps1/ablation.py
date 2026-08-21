"""Ablation ladder from policy 0.

Policy 0 is run 2026-07-10_20-59-17 -- the ONNX on hardware matches its
checkpoint byte for byte. Branch `policy0-reference` reproduces it with an empty
recursive diff on env.yaml and agent.yaml, and it walks: track_linear_velocity
2.60 and stance_contacts_mean 2.81 at iteration 600, settling at 3.0 / 2.89.
Runs that fail sit at 1.95 and 3.6-3.8, so one hour separates the two.

  RHPS1_ABLATION=p0            policy 0's configuration on today's code
  RHPS1_ABLATION=p0+rand       ... plus one rung
  RHPS1_ABLATION=p0+rand+obs   ... cumulative, in ladder order

Unset leaves the configuration untouched. Applying every rung in LADDER order
must land back on the untouched configuration -- asserted by
scripts/tools/check_ablation_ladder.py, which is what keeps the ladder honest:
a deviation missing from every rung would otherwise never be tested.

`p0` here is not the reference tree: nine reward functions were rewritten since
July. They turn out to be reachable anyway -- each grew parameters with a
neutral setting, and the baseline uses them (see flat_support and
standing_single_support below). Comparing effective parameters rather than the
dumped yaml is what found them: the yaml records only what a RewardTermCfg sets
explicitly, so a signature default that changed is invisible there. Anything
still not reachable is listed in check_ablation_ladder.py with its reason.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

ENV_VAR = "RHPS1_ABLATION"

# Cumulative order. Value first: the two rungs the real robot cannot do without
# come before the ones that only shape the gait, so an interrupted ladder still
# answers the questions that matter. `static` is last because it is both the
# most suspect single item -- 0.4 standing envs against policy 0's 0.1 -- and
# the cheapest to act on.
LADDER = ("rand", "obs", "knee", "feet", "prox", "pose", "mirror", "static")


def selection() -> list[str] | None:
  """Steps requested, or None when the variable is unset."""
  raw = os.environ.get(ENV_VAR, "").strip()
  if not raw:
    return None
  steps = [s for s in raw.split("+") if s]
  if not steps or steps[0] != "p0":
    raise ValueError(f"{ENV_VAR} must start with 'p0', got {raw!r}")
  # `fs@-3.24` runs the fs rung with flat_support at that weight, so a sweep is
  # one string per point and the results table names its own weight.
  _INLINE_WEIGHTS.clear()
  bare = []
  for s in steps[1:]:
    if "@" in s:
      name, _, w = s.partition("@")
      if name not in RUNG_TERM:
        raise ValueError(f"{ENV_VAR}: '{name}' owns no reward weight")
      _INLINE_WEIGHTS[RUNG_TERM[name]] = float(w)
      bare.append(name)
    else:
      bare.append(s)
  steps = [steps[0]] + bare
  unknown = set(steps[1:]) - set(DELTAS)
  if unknown:
    raise ValueError(
      f"{ENV_VAR}: unknown steps {sorted(unknown)}, have {sorted(DELTAS)}"
    )
  return steps


def mirror_enabled() -> bool:
  """Symmetry augmentation is a deviation: policy 0 trained without it."""
  steps = selection()
  return True if steps is None else "mirror" in steps


##
# Curriculum functions policy 0 used and that were deleted since. Kept local so
# reverting the config does not reach into shared mdp code.
##


def air_time_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_name: str,
  stages: list[dict],
) -> torch.Tensor:
  del env_ids
  current = stages[0]["value"]
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      current = stage["value"]
  env.reward_manager.get_term_cfg(reward_name).params[param_name] = current
  return torch.tensor([current])


def standing_envs_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[dict],
) -> torch.Tensor:
  del env_ids
  value = stages[0]["value"]
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      value = stage["value"]
  env.command_manager.get_term(command_name).cfg.rel_standing_envs = value
  return torch.tensor([value])


##
# Baseline
##

EXTRA_EVENTS = ("actuator_gains", "link_com", "link_inertia", "sensor_bias")

PROXIMITY_TERMS = (
  "arm_torso_proximity",
  "knee_proximity",
  "leg_proximity",
  "shoulder_body_proximity",
  "shoulder_chest_proximity",
  "wrist_thigh_proximity",
)

PROXIMITY_SENSORS = (
  "knee_proximity",
  "arm_torso_proximity",
  "shoulder_chest_proximity",
  "shoulder_body_proximity",
  "wrist_thigh_proximity",
)

# Policy 0's gap: the three leg links at 12 mm, not the eight QP-pair links at
# 25 mm. Widening it changes contact detection, so it belongs to `prox`.
_P0_GAP = {r"^rhps1_collision_[LR]_(CROTCH_P|KNEE_P|ANKLE_R)_LINK$": 0.012}

_P0_POSE_STD = {
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


def _revert_to_policy0(cfg: ManagerBasedRlEnvCfg) -> None:
  r = cfg.rewards

  # -- rewards -------------------------------------------------------------
  r["angular_momentum"].weight = -0.2
  # flat_support_penalty is a class now, but every extension it grew has a
  # neutral setting, so July's body is reachable after all -- the first p0 rung
  # ran with corner_tolerance 0.001 and measured something else entirely.
  #   corner_tolerance 0  -> contacts = (found > 0), the solver's own detection,
  #                          instead of corner height within a 1 mm band
  #   change_gain      0  -> drops the corner-loss/gain term
  #   standing_thresh -1  -> `standing` is never true, so an unloaded foot is
  #                          not charged a full deficit at zero command
  #   load_threshold   0  -> `loaded` collapses to `in_contact`
  # What is left is sum(square(deficit) * in_contact): July's line for line.
  r["flat_support"].weight = -2.4
  r["flat_support"].params["corner_tolerance"] = 0.0
  r["flat_support"].params["change_gain"] = 0.0
  r["flat_support"].params["standing_threshold"] = -1.0
  r["flat_support"].params["load_threshold"] = 0.0
  r["foot_clearance"].weight = -4.0
  r["foot_slip"].weight = -0.3
  r["impact_vel"].weight = -0.5
  r["impact_vel"].params["limit"] = 0.1
  # grace_period 0 makes this exactly July's
  # cost = (one_foot + 4*no_feet) * standing. At 1.5 s the penalty simply does
  # not apply for the first 1.5 s of every standing episode.
  r["standing_single_support"].weight = -4.0
  r["standing_single_support"].params["grace_period"] = 0.0
  # air_time weight and threshold_max are driven by curriculum below; these are
  # the step-0 values it starts from.
  r["air_time"].weight = 2.0
  r["air_time"].params["power"] = 2.0
  r["air_time"].params["touchdown_cost"] = 0.15
  r["air_time"].params["threshold_max"] = 0.2

  r["leg_proximity"].weight = -1.0
  r["leg_proximity"].params["min_dist"] = 0.01

  r["pose"].params["std_walking"] = dict(_P0_POSE_STD)

  r["ankle_pitch_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2,
    weight=-0.0002,
    params={
      "actuator_pattern": r"^[LR]_ANKLE_P$",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  r["ankle_roll_torque"] = RewardTermCfg(
    func=mdp.joint_effort_l2,
    weight=-0.002,
    params={
      "actuator_pattern": r"^[LR]_ANKLE_R$",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # Terms policy 0 carried that were dropped since. min_foot_height paid zero
  # for standing and punished a short lift -- restored anyway: the baseline is
  # policy 0, defects included, because that is what walked.
  r["min_foot_height"] = RewardTermCfg(
    func=mdp.swing_foot_height,
    weight=-5.0,
    params={
      "min_height": 0.08,
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )
  r["flat_touchdown"] = RewardTermCfg(
    func=mdp.flat_touchdown_penalty,
    weight=-1.8,
    params={
      "sensor_name": "feet_ground_contact_split",
      "required_contacts_per_foot": 4,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  # Zero weight, inert, but it keeps the diff against policy 0 empty.
  r["action_acc_l2"] = RewardTermCfg(func=mdp.action_acc_l2, weight=0.0)

  for name in PROXIMITY_TERMS:
    if name != "leg_proximity":  # policy 0 carried this one, at -1.0
      r.pop(name, None)

  # -- events --------------------------------------------------------------
  for name in EXTRA_EVENTS:
    cfg.events.pop(name, None)
  cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)

  # -- observations --------------------------------------------------------
  actor = cfg.observations["actor"].terms
  actor["actions"].func = mdp.last_action
  actor["actions"].history_length = 0
  actor["base_lin_vel"].func = mdp.base_lin_vel
  actor["base_lin_vel"].noise = None
  actor["joint_pos"].params.pop("biased", None)
  actor["joint_vel"].func = mdp.joint_vel_rel
  actor["joint_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)
  actor["joint_vel"].params.pop("encoder_noise", None)
  actor["projected_gravity"].func = mdp.projected_gravity

  critic = cfg.observations["critic"].terms
  critic.pop("foot_height_scan", None)
  critic.pop("joint_torques", None)
  critic["joint_pos"].noise = Unoise(n_min=-0.01, n_max=0.01)
  critic["joint_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)
  critic["projected_gravity"].func = mdp.projected_gravity

  # -- commands and curriculum --------------------------------------------
  # 0.2 falling to 0.1, not 0.4 held: policy 0 steady-state had a quarter of
  # today's standing envs.
  cfg.commands["twist"].rel_standing_envs = 0.2
  cfg.curriculum["standing_envs"] = CurriculumTermCfg(
    func=standing_envs_curriculum,
    params={
      "command_name": "twist",
      "stages": [{"step": 0, "value": 0.2}, {"step": 24000, "value": 0.1}],
    },
  )
  cfg.curriculum["air_time"] = CurriculumTermCfg(
    func=air_time_curriculum,
    params={
      "reward_name": "air_time",
      "param_name": "threshold_max",
      "stages": [
        {"step": 0, "value": 0.1},
        {"step": 24000, "value": 0.3},
        {"step": 96000, "value": 0.5},
      ],
    },
  )
  cfg.curriculum["air_time_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "air_time",
      "weight_stages": [
        {"step": 0, "weight": 2.0},
        {"step": 24000, "weight": 5.0},
      ],
    },
  )
  cfg.curriculum["velocity_damper"] = CurriculumTermCfg(
    func=mdp.velocity_damper_progress,
    params={"start_step": 360000, "end_step": 612000},
  )

  # -- scene ---------------------------------------------------------------
  ent = cfg.scene.entities["robot"]
  ent.collisions[0].gap = dict(_P0_GAP)
  cfg.scene.sensors = tuple(
    s for s in cfg.scene.sensors if getattr(s, "name", None) not in PROXIMITY_SENSORS
  )

  # Knee effort limit 100 N.m, and the action scale that encodes it. The L/R
  # split is representation only -- policy 0 had one `.*_KNEE_P` group, same
  # velocity limits.
  for act in ent.articulation.actuators:
    if any("KNEE_P" in e for e in act.target_names_expr):
      act.effort_limit = 100.0
  scale = cfg.actions["joint_pos"].scale
  for j in ("L_KNEE_P", "R_KNEE_P"):
    scale.pop(j, None)
  scale[r".*_KNEE_P"] = 0.0075


##
# Rungs. Cumulative, one deviation each, so the cost of a change is read
# directly rather than inferred from a run carrying all of them.
##


def _rand(cfg, full) -> None:
  """Domain randomisation and the sensor models it goes with.

  Load-bearing for transfer, not just robustness margin: restoring these ranges
  is what closed the train/play gap. First rung for that reason.
  """
  for name in EXTRA_EVENTS:
    if name in full["events"]:
      cfg.events[name] = full["events"][name]
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

  actor = cfg.observations["actor"].terms
  actor["base_lin_vel"].func = mdp.base_lin_vel_biased
  actor["base_lin_vel"].noise = full["actor"]["base_lin_vel"].noise
  actor["joint_pos"].params["biased"] = True
  actor["joint_vel"].func = mdp.joint_vel_encoder_finite_difference
  actor["joint_vel"].noise = None
  actor["joint_vel"].params["encoder_noise"] = 5e-05
  actor["projected_gravity"].func = mdp.projected_gravity_biased

  critic = cfg.observations["critic"].terms
  critic["joint_pos"].noise = None
  critic["projected_gravity"].func = mdp.projected_gravity_biased


def _obs(cfg, full) -> None:
  """executed_action plus 5 frames of history: observation 126 -> 246.

  The actuator carries hidden state (EMA, finite-difference qd*) and nothing in
  policy 0's observation did. Required for the deployed stack to see what
  training saw.
  """
  actor = cfg.observations["actor"].terms
  actor["actions"].func = mdp.executed_action
  actor["actions"].history_length = 5
  critic = cfg.observations["critic"].terms
  if "joint_torques" in full["critic"]:
    critic["joint_torques"] = full["critic"]["joint_torques"]
  if "foot_height_scan" in full["critic"]:
    critic["foot_height_scan"] = full["critic"]["foot_height_scan"]


def _knee(cfg, full) -> None:
  """Knee effort limit 100 -> 70 N.m, with the matching action scale.

  The real knee cannot do 100. Known suspect: at 70 the projection saturated at
  91% and the policy traded advancing for cadence. Late in the ladder so
  everything below it is already banked.
  """
  ent = cfg.scene.entities["robot"]
  for act in ent.articulation.actuators:
    if any("KNEE_P" in e for e in act.target_names_expr):
      act.effort_limit = 70.0
  scale = cfg.actions["joint_pos"].scale
  scale.pop(r".*_KNEE_P", None)
  for j in ("L_KNEE_P", "R_KNEE_P"):
    scale[j] = 0.00525


def _feet(cfg, full) -> None:
  """Foot shaping, recalibrated as one block because it was tuned as one.

  Drops min_foot_height -- the term that paid zero for standing and punished a
  short lift -- and flat_touchdown, whose job the rewritten flat_support does.
  """
  r = cfg.rewards
  r["flat_support"].weight = -11.0
  # The extensions the baseline neutralises, back as the config really had them.
  # Restore by deleting when the config left them implicit, so the dumped yaml
  # matches too -- setting the signature default would still show as a new key.
  for term, keys in (
    ("flat_support",
     ("corner_tolerance", "change_gain", "standing_threshold", "load_threshold")),
    ("standing_single_support", ("grace_period",)),
  ):
    for p in keys:
      if p in full["rewards"][term].params:
        r[term].params[p] = full["rewards"][term].params[p]
      else:
        r[term].params.pop(p, None)
  r["foot_clearance"].weight = -10.0
  r["foot_slip"].weight = -0.5
  r["impact_vel"].weight = -2.0
  r["impact_vel"].params["limit"] = 0.15
  r["standing_single_support"].weight = -6.0
  r["air_time"].params["power"] = 1.0
  r["air_time"].params["touchdown_cost"] = 0.0
  r.pop("min_foot_height", None)
  r.pop("flat_touchdown", None)


##
# Decomposition rungs. `feet` and `obs` each bundle several changes and each
# broke walking as a block, so these split them into one change per run.
#
# Weights are overridable from the environment -- RHPS1_W_flat_support=-12.4 --
# because a measurement change rescales a penalty and the weight that goes with
# it has to be found rather than assumed. See docs/reward_tuning.md.
##


# Which reward term a decomposition rung owns, for the `rung@weight` syntax.
RUNG_TERM = {
  "fs": "flat_support",
  "fsct": "flat_support",
  "fscg": "flat_support",
  "fsload": "flat_support",
  "air": "air_time",
  "sss": "standing_single_support",
  "imp": "impact_vel", "tq": "joint_torques_l2",
  "swt": "foot_swing_height", "mfhr": "min_foot_height",
  "airtc": "air_time", "airT": "air_time",
}

_INLINE_WEIGHTS: dict[str, float] = {}


def _w(cfg, term: str, default: float) -> None:
  """Set a reward weight. `rung@weight` wins, then RHPS1_W_<term>, then default."""
  if term in _INLINE_WEIGHTS:
    cfg.rewards[term].weight = _INLINE_WEIGHTS[term]
    return
  raw = os.environ.get(f"RHPS1_W_{term}")
  cfg.rewards[term].weight = float(raw) if raw else default


def _fs(cfg, full) -> None:
  """flat_support: the corrected corner measurement, weight tunable.

  corner_tolerance counts corners by height inside a 1 mm band instead of by the
  solver's contact detection -- the better measurement, since the sole is
  parallel to within 16 um yet only 2.15 of 4 corners registered. It shrinks the
  penalty, so -2.4 no longer means what it did; -11 is roughly parity by
  realized value, which is why this rung is weight-tunable rather than fixed.
  """
  for p in ("corner_tolerance", "change_gain", "standing_threshold", "load_threshold"):
    if p in full["rewards"]["flat_support"].params:
      cfg.rewards["flat_support"].params[p] = full["rewards"]["flat_support"].params[p]
    else:
      cfg.rewards["flat_support"].params.pop(p, None)
  _w(cfg, "flat_support", -11.0)


def _fsct(cfg, full) -> None:
  """flat_support: corner_tolerance only, the measurement fix on its own.

  The one Leo asked about. Counts corners by height inside a 1 mm band instead
  of by solver detection -- the sole is parallel to within 16 um in the default
  pose yet only 2.15 of 4 corners registered. Weight tunable.
  """
  cfg.rewards["flat_support"].params["corner_tolerance"] = 0.001
  _w(cfg, "flat_support", -2.4)


def _fscg(cfg, full) -> None:
  """flat_support: change_gain only -- a new penalty, not a measurement fix.

  Charges corners lost between two steps and credits corners gained, on feet
  loaded in both. Nothing in July did this, so it cannot be calibrated against
  a July value; it is a design choice to accept or drop.
  """
  cfg.rewards["flat_support"].params["change_gain"] = 1.0
  _w(cfg, "flat_support", -2.4)


def _fsload(cfg, full) -> None:
  """flat_support: the load and standing thresholds only.

  `loaded` becomes force > 140 N instead of any corner in contact, and a foot
  that is unloaded at zero command is charged the full deficit rather than
  nothing.
  """
  cfg.rewards["flat_support"].params["load_threshold"] = 140.0
  cfg.rewards["flat_support"].params["standing_threshold"] = 0.1
  _w(cfg, "flat_support", -2.4)


def _air(cfg, full) -> None:
  """air_time: power 2 -> 1 and touchdown_cost 0.15 -> 0, weight tunable.

  Largest realized shift of the block after min_foot_height: the landing bonus
  pays 6.4x more per unit. A linear payout with no touchdown cost makes a short
  hop profitable, which is the mechanism behind "the foot lifts and never lands".
  """
  cfg.rewards["air_time"].params["power"] = 1.0
  cfg.rewards["air_time"].params["touchdown_cost"] = 0.0
  _w(cfg, "air_time", 2.0)


def _mfh(cfg, full) -> None:
  """Drop min_foot_height, as the feet block does.

  Worth its own run: it is the single biggest realized change in the block
  (-0.303 per step removed). It was called a trap -- zero for standing, a
  penalty for a short lift -- but it is also the only strong pressure to clear
  8 cm, and the block that removes it lifts 65% less.
  """
  cfg.rewards.pop("min_foot_height", None)


def _sss(cfg, full) -> None:
  """standing_single_support: grace_period back, weight tunable.

  1.5 s of grace plus -6 leaves a realized -0.011 against -0.065: the penalty
  all but disappears.
  """
  if "grace_period" in full["rewards"]["standing_single_support"].params:
    cfg.rewards["standing_single_support"].params["grace_period"] = full["rewards"][
      "standing_single_support"
    ].params["grace_period"]
  else:
    cfg.rewards["standing_single_support"].params.pop("grace_period", None)
  _w(cfg, "standing_single_support", -6.0)


def _imp(cfg, full) -> None:
  """impact_vel: limit 0.1 -> 0.15, weight tunable."""
  cfg.rewards["impact_vel"].params["limit"] = 0.15
  _w(cfg, "impact_vel", -2.0)


def _hist(cfg, full) -> None:
  """Action history 0 -> 5 frames, nothing else. Observation 126 -> 246."""
  cfg.observations["actor"].terms["actions"].history_length = 5


def _exec(cfg, full) -> None:
  """last_action -> executed_action, nothing else.

  Only meaningful with the torque projection: it reports what the projection
  executed, and torque_feasibility_ratio is None here, so pair it with `proj`.
  """
  cfg.observations["actor"].terms["actions"].func = mdp.executed_action


def _proj(cfg, full) -> None:
  """Torque feasibility projection at ratio 1.0, the way `exec` was validated."""
  for act in cfg.scene.entities["robot"].articulation.actuators:
    act.torque_feasibility_ratio = 1.0


def _ctorque(cfg, full) -> None:
  """Critic sees joint torques."""
  if "joint_torques" in full["critic"]:
    cfg.observations["critic"].terms["joint_torques"] = full["critic"]["joint_torques"]


def _cscan(cfg, full) -> None:
  """Critic sees the per-foot height scan."""
  if "foot_height_scan" in full["critic"]:
    cfg.observations["critic"].terms["foot_height_scan"] = full["critic"][
      "foot_height_scan"
    ]


def _prox(cfg, full) -> None:
  """Five proximity sensors and their penalties, plus the wider collision gap.

  Hardware protection, matched to the QP's collision margins. The gap widening
  goes here because it is what the sensors measure against.
  """
  # Restored whole rather than appended: the proximity sensors are interleaved
  # with the scans and the manager keys observations by index.
  cfg.scene.sensors = copy.deepcopy(full["sensors"])
  for name in PROXIMITY_TERMS:
    if name in full["rewards"]:
      cfg.rewards[name] = full["rewards"][name]  # leg_proximity goes to -2.0
  cfg.scene.entities["robot"].collisions[0].gap = copy.deepcopy(full["gap"])


def _pose(cfg, full) -> None:
  """Looser pose targets, and the two objective softenings that came with them."""
  cfg.rewards["pose"].params["std_walking"] = copy.deepcopy(
    full["rewards"]["pose"].params["std_walking"]
  )
  cfg.rewards["angular_momentum"].weight = -0.3
  cfg.rewards.pop("ankle_roll_torque", None)
  cfg.rewards.pop("ankle_pitch_torque", None)


def _mirror(cfg, full) -> None:
  """Symmetry augmentation. Agent-side, via mirror_enabled(); no env change."""


def _static(cfg, full) -> None:
  """Drop policy 0's four curricula and hold rel_standing_envs at 0.4.

  Last and most suspect: policy 0 walked with a tenth of the envs standing,
  today's config holds four tenths, and air_time's weight and threshold reach
  their end values at step 0 instead of ramping.
  """
  for name in ("standing_envs", "air_time", "air_time_weight", "velocity_damper"):
    cfg.curriculum.pop(name, None)
  cfg.commands["twist"].rel_standing_envs = 0.4
  r = cfg.rewards
  r["air_time"].weight = 5.0
  r.pop("action_acc_l2", None)


# Decomposition rungs are outside LADDER: they split blocks that LADDER already
# covers, so including them would double-apply and break the completeness check.
##
# Repair rungs. Everything above restores or isolates what policy 0 already had;
# these change the objective on purpose, on the diagnosis in docs/reward_map.md.
# Untested by anyone -- judged on the same four criteria as the rest.
##


def _swt(cfg, full) -> None:
  """foot_swing_height: bring the target down to where the policy lives.

  Right shape already -- peak per swing, charged once at landing -- but the
  0.15 m target sits two orders above the operating point, which pins the
  squared relative error near 1.0 whatever the gait does. A term stuck at its
  ceiling is a per-landing tax, and the only way to lower a tax on landing is to
  land less.
  """
  t = float(os.environ.get("RHPS1_SWT_TARGET", "0.05"))
  cfg.rewards["foot_swing_height"].params["target_height"] = t
  _w(cfg, "foot_swing_height", -5.0)


def _mfhr(cfg, full) -> None:
  """min_foot_height: fade the charge in over the first 60 ms of flight.

  Removes the liftoff discontinuity without removing the term. See
  swing_foot_height's docstring for the barrier this dissolves.
  """
  r = cfg.rewards.get("min_foot_height")
  if r is not None:
    r.params["ramp_s"] = float(os.environ.get("RHPS1_MFH_RAMP", "0.06"))
    _w(cfg, "min_foot_height", -5.0)


def _fclr(cfg, full) -> None:
  """Drop foot_clearance: it charges |z - 0.15| * horizontal foot speed.

  The foot is never near 0.15 m, so the first factor is a near-constant ~0.10
  and what the term actually taxes is swinging the foot forward. Named for
  clearance, priced as a penalty on stepping.
  """
  cfg.rewards.pop("foot_clearance", None)


def _airtc(cfg, full) -> None:
  """air_time: drop touchdown_cost, keep the square.

  touchdown_cost puts break-even at t = threshold_max * sqrt(0.15), so landing
  short of that is punished. Removing it makes the payout non-negative
  everywhere while the square keeps paying long flights disproportionately --
  the amplitude incentive that linearising (`air`) threw away.
  """
  cfg.rewards["air_time"].params["touchdown_cost"] = 0.0
  _w(cfg, "air_time", 2.0)


def _airT(cfg, full) -> None:
  """air_time: stop the curriculum walking threshold_max away from the gait.

  The stages take it 0.1 -> 0.3 -> 0.5, moving break-even 0.039 -> 0.116 ->
  0.194 s while nothing pulls the flight time after it. Pin it instead.
  """
  cfg.curriculum.pop("air_time", None)
  cfg.rewards["air_time"].params["threshold_max"] = float(
    os.environ.get("RHPS1_AIR_TMAX", "0.15"))
  _w(cfg, "air_time", 2.0)


def _lift(cfg, full) -> None:
  """Raise the swing foot and lengthen the flight, on measured numbers.

  reward_audit on policy 0's own checkpoint: swing foot z p50 5.6 mm, p90 8.7,
  max 16.6; landing flight time p50 0.178 s, p90 0.238 s.

  Both height terms sit at their ceiling at that operating point, so neither
  carries gradient:
    min_foot_height   cost 0.063 of a 0.069 maximum -- 92%, a flat tax on being
                      airborne, which is also a tax on the air time we want.
                      min_height 0.08 -> 0.02 restores range (cost 0.063 ->
                      0.012 across the same gait) and ramp_s dissolves the
                      liftoff cliff.
    foot_swing_height squared *relative* error saturates at 1.0 for any target
                      well above the peak: dropping 0.15 -> 0.02 moves the cost
                      only 11%. Set 0.03 anyway -- it is free gradient -- but do
                      not expect it to carry the change.

  air_time is worse than idle: the curriculum walks threshold_max 0.1 -> 0.3 ->
  0.5, and break-even is threshold_max*sqrt(touchdown_cost). At 0.5 that is
  0.194 s, above policy 0's median landing, so the median landing is paid
  -0.024. Pinned at 0.25 the median pays +0.354 and the p90 +0.752, both on the
  rising part of the curve. touchdown_cost stays at 0.15: with break-even down
  at 0.097 s it no longer punishes normal landings, and it is what stops the
  cadence exploit that linearising the payout produced.

  Knee stays at 100 N.m and torque_limit_margin is untouched: the point is to
  lift the foot without spending the property policy 0 was kept for.
  """
  r = cfg.rewards
  mfh = r.get("min_foot_height")
  if mfh is not None:
    mfh.params["min_height"] = float(os.environ.get("RHPS1_MFH_MIN", "0.02"))
    mfh.params["ramp_s"] = float(os.environ.get("RHPS1_MFH_RAMP", "0.06"))
  r["foot_swing_height"].params["target_height"] = float(
    os.environ.get("RHPS1_SWT_TARGET", "0.03"))
  cfg.curriculum.pop("air_time", None)
  r["air_time"].params["threshold_max"] = float(
    os.environ.get("RHPS1_AIR_TMAX", "0.25"))


def _stride(cfg, full) -> None:
  """Longer steps, by paying for a slower cycle rather than a faster foot.

  Measured on policy 0: flight 0.178 s, per-foot air fraction 0.43, speed
  0.136 m/s -- a 0.414 s cycle, so a 2.8 cm step at 2.4 Hz. Step length is
  speed / (2 * cycle frequency) and the speed is commanded, so the policy is
  free to meet it with a short fast step or a long slow one. Nothing in the
  objective pays for the stride, and four action-rate terms plus foot_clearance
  tax it, so it picks the shuffle. Correctly.

  The one term that pays for cycle time is air_time, capped at threshold_max:
  0.25 buys a 3.9 cm step, 0.5 buys 7.9 cm. The curriculum already aimed at 0.5
  and could not get there because touchdown_cost put break-even at
  0.387*threshold_max = 0.194 s, above the gait. Remove the forfeit instead of
  lowering the cap.

  No cadence exploit: with power 2, two 0.1 s flights pay 2*(0.2)^2 = 0.08 and
  one 0.2 s flight pays (0.4)^2 = 0.16, so amplitude wins by construction. That
  is what `air` broke by linearising -- at power 1 both pay 0.40 and the policy
  goes indifferent. The square was the guard, not the forfeit.

  feet_distance rides along: both policy 0 (0.1944) and the current run (0.2003)
  sit on max_distance = 0.20, and while the penalty there is small enough not to
  bind today, an 8 cm step would put the mean well past it.
  """
  _lift(cfg, full)
  r = cfg.rewards
  r["air_time"].params["threshold_max"] = float(
    os.environ.get("RHPS1_AIR_TMAX", "0.5"))
  r["air_time"].params["touchdown_cost"] = 0.0
  r["feet_distance"].params["max_distance"] = 0.30


def _tq(cfg, full) -> None:
  """Hold the torque down while the stride grows, by widening the barrier.

  torque_limit_margin is clamp((|tau|/limit - soft_ratio)/(1 - soft_ratio), 0)^2
  summed over joints, so it charges nothing below 80% of the limit. Its raw cost
  is 3.06 against a mean ratio of 0.357: roughly three joints ride at the limit
  while the other thirty cruise at a third of it. The load is concentrated.

  That rules out raising its weight. A joint pinned at the limit has normalized
  capped at 1.0 and excess capped at 1.0, so more weight buys a bigger constant
  with no gradient -- the same saturation pathology as foot_swing_height at an
  unreachable target. Lower soft_ratio instead: at 0.6 a joint at 0.8 goes from
  paying nothing to paying 0.25, while a joint at 1.0 still pays exactly 1.0.
  Gradient appears below the cap and nothing escalates at it.

  joint_torques_l2 doubles as the global counterpart. It is the literal
  "minimise the torque" term, and at 4.7% of the penalty budget it is small
  enough that doubling it is the modest change it looks like.
  """
  cfg.rewards["torque_limit_margin"].params["soft_ratio"] = float(
    os.environ.get("RHPS1_SOFT_RATIO", "0.6"))
  _w(cfg, "joint_torques_l2", -2e-5)


def _nodamp(cfg, full) -> None:
  """Drop the velocity damper entirely. Leo's call, 2026-08-21.

  The curriculum ramps velocity_damper_progress 0 -> 1 between iterations 7500
  and 12750, and progress 0 is a no-op, so a three-hour run never reaches it
  anyway -- removing it changes nothing about what comes out of one. What it
  changes is the intent: the damper is a projection of the position target that
  matches the mc_rtc KinematicsConstraint, so this makes it explicit that we
  train on a different plant than the robot runs.

  The QP applies it at deployment regardless, and the mismatch is largest
  exactly where `stride` aims: di is 0.4 of the joint *range*, so a longer step
  walks further into the damped zone than the 2.8 cm shuffle ever did. Recorded
  as a rung rather than by deleting the curriculum, so it reads as a choice.
  """
  cfg.curriculum.pop("velocity_damper", None)


DECOMPOSED = {
  "fs": _fs, "fsct": _fsct, "fscg": _fscg, "fsload": _fsload, "air": _air, "mfh": _mfh, "sss": _sss, "imp": _imp,
  "hist": _hist, "exec": _exec, "proj": _proj,
  "ctorque": _ctorque, "cscan": _cscan,
  "lift": _lift, "stride": _stride, "tq": _tq, "nodamp": _nodamp,
  "swt": _swt, "mfhr": _mfhr, "fclr": _fclr, "airtc": _airtc, "airT": _airT,
}

DELTAS = {
  **DECOMPOSED,
  "rand": _rand,
  "obs": _obs,
  "knee": _knee,
  "feet": _feet,
  "prox": _prox,
  "pose": _pose,
  "mirror": _mirror,
  "static": _static,
}


def apply_env(cfg: ManagerBasedRlEnvCfg) -> None:
  """Revert to policy 0, then re-apply the requested deviations in order."""
  steps = selection()
  if steps is None:
    return
  # Snapshot first: rungs that re-add something restore it from here rather
  # than redeclaring it, so a rung cannot drift from what the config says.
  full = {
    "rewards": copy.deepcopy(cfg.rewards),
    "events": copy.deepcopy(cfg.events),
    "actor": copy.deepcopy(cfg.observations["actor"].terms),
    "critic": copy.deepcopy(cfg.observations["critic"].terms),
    "sensors": copy.deepcopy(cfg.scene.sensors),
    "gap": copy.deepcopy(cfg.scene.entities["robot"].collisions[0].gap),
  }
  _revert_to_policy0(cfg)
  requested = steps[1:]
  # Ladder rungs in ladder order, so a typo in the order cannot change the run.
  for step in LADDER:
    if step in requested:
      DELTAS[step](cfg, full)
  # Decomposition rungs in the order given: they are read as a sequence
  # ("history, then executed_action, then the critic terms").
  for step in requested:
    if step in DECOMPOSED:
      DELTAS[step](cfg, full)
