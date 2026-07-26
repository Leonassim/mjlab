"""RHPS1-specific rsl-rl extensions: held exploration noise, mirror symmetry,
and torque-feasibility guidance.

Everything here plugs into stock rsl-rl through its string-resolved config
hooks (``distribution_cfg.class_name``, ``symmetry_cfg.data_augmentation_func``,
``algorithm.class_name``), so no rsl-rl fork is needed and upstream mjlab
files are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from mjlab.rl import RslRlPpoAlgorithmCfg

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv


class HeldNoiseGaussianDistribution(GaussianDistribution):
  """Gaussian policy noise held constant for ``hold_steps`` control steps.

  White per-step noise on position targets does not integrate: the limb stays
  in a tube of radius std*action_scale around the policy mean while every
  resample demands a torque spike of kp * std * scale (hundreds of N.m at
  RHPS1 gains -- the actuators saturate on noise alone). Holding each noise
  draw for ``hold_steps`` steps moves the same variance to low frequency:
  the limb has time to track the offset (exploration becomes visible motion)
  and the torque spike happens once per window instead of every step.

  The per-step marginal stays N(mean, std), so log-prob, entropy, KL and the
  ONNX export path are untouched; only ``sample()`` changes. This is the
  feature-less special case of gSDE (Raffin et al., 2021).

  Rollout detection: buffers are lazily created at the first ``sample()``
  call (always a rollout, sized num_envs). Later calls with a different
  batch size come from the PPO update epochs, where the sampled value is
  discarded -- those fall through to plain white sampling without touching
  the held state.
  """

  def __init__(self, output_dim: int, hold_steps: int = 16, **kwargs) -> None:
    super().__init__(output_dim, **kwargs)
    self.hold_steps = int(hold_steps)
    self._eps: torch.Tensor | None = None
    self._countdown: torch.Tensor | None = None

  def sample(self) -> torch.Tensor:
    dist = self._distribution
    assert dist is not None
    mean = dist.mean
    std = dist.stddev
    batch = mean.shape[0]

    if self._eps is None:
      self._eps = torch.randn(
        batch, self.output_dim, device=mean.device, dtype=mean.dtype
      )
      # Random initial phases so the envs do not resample in lockstep.
      self._countdown = torch.randint(
        1, self.hold_steps + 1, (batch,), device=mean.device
      )
      return mean + std * self._eps

    if self._eps.shape[0] != batch:
      # PPO update epochs re-run the actor with stochastic_output=True on
      # mini-batches and discard the sample; do not disturb the held state.
      return super().sample()

    assert self._countdown is not None
    self._countdown -= 1
    expired = self._countdown <= 0
    if bool(torch.any(expired)):
      num_expired = int(expired.sum())
      self._eps[expired] = torch.randn(
        num_expired, self.output_dim, device=mean.device, dtype=mean.dtype
      )
      self._countdown[expired] = self.hold_steps
    return mean + std * self._eps


# ---------------------------------------------------------------------------
# Mirror symmetry (left-right reflection about the sagittal x-z plane).
# ---------------------------------------------------------------------------
#
# Sign conventions (verified against the mirror-symmetric RHPS1 keyframe):
# pitch joints keep their value under the mirror, roll and yaw joints flip
# sign. Midline joints (CHEST_*, HEAD_*) map onto themselves with the same
# rule. World vectors: y components of linear quantities flip; x and z
# components of angular quantities flip.

_VEC3_LIN_SIGN = (1.0, -1.0, 1.0)  # lin vel, gravity
_VEC3_ANG_SIGN = (-1.0, 1.0, -1.0)  # ang vel
_VEC3_CMD_SIGN = (1.0, -1.0, -1.0)  # (vx, vy, wz)


def _mirror_joint_name(name: str) -> str:
  if name.startswith("L_"):
    return "R_" + name[2:]
  if name.startswith("R_"):
    return "L_" + name[2:]
  return name


def _joint_sign(name: str) -> float:
  return -1.0 if name.endswith(("_R", "_Y")) else 1.0


def _joint_perm_sign(joint_names: list[str]) -> tuple[list[int], list[float]]:
  index = {n: i for i, n in enumerate(joint_names)}
  perm, sign = [], []
  for name in joint_names:
    partner = _mirror_joint_name(name)
    if partner not in index:
      raise ValueError(f"No mirror partner for joint '{name}'")
    perm.append(index[partner])
    sign.append(_joint_sign(name))
  return perm, sign


def _tile_block(
  block_perm: list[int], block_sign: list[float], total: int, offset: int
) -> tuple[list[int], list[float]]:
  """Tile a block permutation across a (possibly history-stacked) flat term."""
  size = len(block_perm)
  if total % size != 0:
    raise ValueError(f"Term of size {total} not divisible by block size {size}")
  perm, sign = [], []
  for rep in range(total // size):
    base = offset + rep * size
    perm.extend(base + p for p in block_perm)
    sign.extend(block_sign)
  return perm, sign


# Per-term mirror rules. Terms absent from this table are left unchanged,
# which is only correct for terms that are symmetric by construction
# (height_scan on flat terrain). Keep use_data_augmentation=False unless
# every critic term has an exact rule.
_TERM_RULES = {
  "base_lin_vel": ([0, 1, 2], list(_VEC3_LIN_SIGN)),
  "projected_gravity": ([0, 1, 2], list(_VEC3_LIN_SIGN)),
  "base_ang_vel": ([0, 1, 2], list(_VEC3_ANG_SIGN)),
  "command": ([0, 1, 2], list(_VEC3_CMD_SIGN)),
  "foot_air_time": ([1, 0], [1.0, 1.0]),  # (left, right) -> swap
  "foot_contact": ([1, 0], [1.0, 1.0]),
  # 2 feet x (fx, fy, fz): swap feet, flip y.
  "foot_contact_forces": ([3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]),
  # [sin_L, cos_L, sin_R, cos_R]: mirroring swaps which foot is "left" --
  # plain block swap, no extra sign flip needed since sin_R/cos_R are
  # already the negation of sin_L/cos_L by construction (half-cycle offset).
  "gait_phase": ([2, 3, 0, 1], [1.0, 1.0, 1.0, 1.0]),
  # (left, right) terrain-relative heights -> swap. Height is a signed
  # scalar about the vertical axis, which a left-right mirror leaves alone,
  # so no sign flip.
  "foot_height_scan": ([1, 0], [1.0, 1.0]),
}
# Per-joint quantities in the robot's joint order: mirrored with the
# left/right joint permutation and per-joint sign flips, same as joint_pos.
# joint_torques belongs here for the same reason joint_vel does -- it is one
# scalar per joint in that identical order (it was missing until 2026-07-25,
# so the critic's mirrored torque history was silently left unpermuted,
# pairing left-leg torques with right-leg states).
_JOINT_SPACE_TERMS = ("joint_pos", "joint_vel", "actions", "joint_torques")

_spec_cache: dict[int, dict] = {}


def _build_specs(env: VecEnv) -> dict:
  """Build (perm, sign) index tensors per observation group and for actions."""
  unwrapped = env.unwrapped
  device = unwrapped.device
  robot = unwrapped.scene["robot"]
  jperm, jsign = _joint_perm_sign(list(robot.joint_names))

  specs: dict = {"groups": {}}
  obs_manager = unwrapped.observation_manager
  for group, term_names in obs_manager.active_terms.items():
    term_dims = obs_manager.group_obs_term_dim[group]
    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for name, dims in zip(term_names, term_dims):
      total = int(np.prod(dims))
      if name in _JOINT_SPACE_TERMS:
        block_perm, block_sign = jperm, jsign
      elif name in _TERM_RULES:
        block_perm, block_sign = _TERM_RULES[name]
      else:
        block_perm, block_sign = list(range(total)), [1.0] * total
      term_perm, term_sign = _tile_block(block_perm, block_sign, total, offset)
      perm.extend(term_perm)
      sign.extend(term_sign)
      offset += total
    specs["groups"][group] = (
      torch.tensor(perm, dtype=torch.long, device=device),
      torch.tensor(sign, device=device),
    )

  # Action vector: concatenation of the action terms' targets in term order.
  action_manager = unwrapped.action_manager
  perm = []
  sign = []
  offset = 0
  for term_name in action_manager.active_terms:
    term = action_manager.get_term(term_name)
    target_names = list(term.target_names)
    block_perm, block_sign = _joint_perm_sign(target_names)
    term_perm, term_sign = _tile_block(block_perm, block_sign, term.action_dim, offset)
    perm.extend(term_perm)
    sign.extend(term_sign)
    offset += term.action_dim
  specs["actions"] = (
    torch.tensor(perm, dtype=torch.long, device=device),
    torch.tensor(sign, device=device),
  )
  return specs


def _get_specs(env: VecEnv) -> dict:
  key = id(env)
  if key not in _spec_cache:
    _spec_cache[key] = _build_specs(env)
  return _spec_cache[key]


def rhps1_mirror(
  env: VecEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Data augmentation function for the rsl-rl Symmetry extension.

  Returns ``[original; mirrored]`` concatenated along the batch dimension for
  whichever of ``obs`` / ``actions`` is provided.
  """
  specs = _get_specs(env)

  obs_out = None
  if obs is not None:
    mirrored = {}
    for key in obs.keys():
      value = obs[key]
      if key in specs["groups"]:
        perm, sign = specs["groups"][key]
        mirrored[key] = value[..., perm] * sign
      else:
        mirrored[key] = value
    obs_out = torch.cat(
      [obs, TensorDict(mirrored, batch_size=obs.batch_size)], dim=0
    )

  actions_out = None
  if actions is not None:
    perm, sign = specs["actions"]
    actions_out = torch.cat([actions, actions[..., perm] * sign], dim=0)

  return obs_out, actions_out


# ---------------------------------------------------------------------------
# Torque-feasibility guidance (direct actor-space auxiliary loss).
# ---------------------------------------------------------------------------


@dataclass
class RhpsPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Adds the torque-guidance fields on top of the shared PPO config.

  Kept as a local subclass rather than extending ``RslRlPpoAlgorithmCfg``
  itself: that dataclass is shared by every task's rl_cfg.py (g1, go1, yam,
  cartpole, tracking), and ``train.py`` forwards its ``asdict()`` straight
  into ``alg_class(**cfg["algorithm"])`` with no per-task filtering. Adding
  fields there would hand ``torque_guidance_coef`` etc. to stock
  ``PPO.__init__`` for every other task and crash on the unexpected keyword.
  Subclassing locally means ``asdict()`` only picks up these extra fields
  for whichever runner config actually instantiates ``RhpsPpoAlgorithmCfg``.
  """

  torque_guidance_coef: float = 0.0
  """Weight for the actor-space torque-feasibility auxiliary loss, once
  warmup has fully ramped in. 0.0 disables the extra optimizer step in
  ``TorqueGuidedPPO.update``."""
  torque_guidance_obs_group: str = "torque_guidance"
  """Observation group holding the per-step regression target (see
  ``mdp.pd_action_guidance_target``)."""
  torque_guidance_obs_term: str = "target"
  """Term name within ``torque_guidance_obs_group`` holding the target."""
  torque_guidance_warmup_updates: int = 0
  """Linearly ramp the effective coefficient from 0 to ``torque_guidance_coef``
  over this many ``update()`` calls. 0 (default) applies the full coefficient
  from the first update. See ``TorqueGuidedPPO`` for why this matters even
  with ``max_action_delta`` capping the target: at iteration 0 essentially
  every env is still falling every episode (the policy hasn't learned to
  balance yet), so without a warmup, most of the batch is still fall-instant
  targets pulling the actor together in the same direction every single
  update -- a consistent bias each step, not a one-off spike, so gradient
  clipping alone does not stop it from accumulating."""


class TorqueGuidedPPO(PPO):
  """PPO plus a supervised auxiliary loss pulling the actor toward the
  torque-feasible action, bypassing the critic/advantage entirely.

  Context: ``mdp.pd_demand_excess`` used to enforce torque feasibility as a
  reward term (weight ramped in by curriculum, up to -8.0). Twice, once the
  weight got large enough to matter, the policy grew timid on the large
  corrective actions needed to catch a stumble -- suppressing exactly the
  recoveries that prevent falls -- and a fall's own chaotic demand then fed
  the same penalty right back in, a runaway loop visible as the ratio metric
  spiking in lockstep with the fall rate. A reward-routed signal has to go
  through the return and GAE before it reaches the policy gradient, so its
  effective strength is entangled with every other reward's scale and with
  the value function's ability to attribute credit -- there is no way to
  make it "just strong enough" without it also reshaping unrelated behavior.

  This term instead regresses the actor's mean action directly onto
  ``mdp.pd_action_guidance_target`` (the action that would have produced the
  actually-applied, clamped torque instead of the unclamped demand): a
  Huber loss, proportional to the actual wasted overshoot, that never
  touches the critic. It runs as one extra epoch over the same rollout,
  after the standard PPO update has already applied its optimizer step for
  this iteration -- deliberately not fused into the surrogate-loss backward,
  so it can't be scaled up or down by entropy/value/symmetry loss weighting
  the way the old reward term was.

  Three attempts before this one all reproduced the same failure at
  different timescales: fell_down and this loss climbing together, only
  slower each time a safeguard was added.
  1. No cap, full coefficient from update 0: collapsed in ~300 iterations
     (most envs still fall every episode that early, and unbounded fall-
     instant targets dominated the batch).
  2. ``max_action_delta`` capped the target, ``torque_guidance_warmup_updates``
     delayed full coefficient to update 1000: collapsed anyway once the
     ramp approached full strength (~iteration 900-1200) -- capping bounds
     one sample's severity, not how many degenerate samples can pull the
     same direction at once, and delaying when full strength arrives doesn't
     help if the batch composition hasn't actually improved by then.
  3. Lowered the coefficient ceiling itself (1.0 -> 0.1): same shape, just
     slower again -- strong evidence this was never really about picking
     the right number.
  The actual fix is in ``_torque_guidance_step``: mask out samples with
  non-positive advantage (a free, already-computed proxy for "state worth
  imitating" -- a fall-bound state's return-to-go is tanked by the
  termination penalty, so its advantage is reliably negative) and use a
  Huber loss instead of MSE so remaining outliers among the kept samples
  don't dominate the batch average either.

  ``torque_guidance_coef=0.0`` (the default) disables the extra step
  entirely, making this identical to stock PPO.
  """

  def __init__(
    self,
    *args,
    torque_guidance_coef: float = 0.0,
    torque_guidance_obs_group: str = "torque_guidance",
    torque_guidance_obs_term: str = "target",
    torque_guidance_warmup_updates: int = 0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.torque_guidance_coef = float(torque_guidance_coef)
    self.torque_guidance_obs_group = torque_guidance_obs_group
    self.torque_guidance_obs_term = torque_guidance_obs_term
    self.torque_guidance_warmup_updates = int(torque_guidance_warmup_updates)
    self._update_count = 0

  def update(self) -> dict[str, float]:
    loss_dict = super().update()
    self._update_count += 1
    coef = self.torque_guidance_coef
    if self.torque_guidance_warmup_updates > 0:
      coef *= min(1.0, self._update_count / self.torque_guidance_warmup_updates)
    if coef > 0.0:
      loss_dict["torque_guidance"] = self._torque_guidance_step(coef)
    return loss_dict

  def _torque_guidance_step(self, coef: float) -> float:
    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, 1)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)

    total_loss = 0.0
    num_batches = 0
    for batch in generator:
      # Advantage sign as a free, already-computed proxy for "was this state
      # trustworthy to imitate": a state on the way to a fall carries a
      # strongly negative advantage (the termination penalty tanks its
      # return-to-go), so it gets excluded here rather than pulling the
      # actor toward whatever chaotic target its q_err implies. Two
      # coefficients (1.0, then 0.1) both eventually reproduced the same
      # fell_down/this-loss runaway once enough of the batch was fall-
      # adjacent -- capping the target bounds one sample's severity but not
      # how many degenerate samples can vote the same direction at once.
      keep = (batch.advantages.squeeze(-1) > 0.0)
      if not bool(torch.any(keep)):
        continue

      target = batch.observations[self.torque_guidance_obs_group]
      if isinstance(target, TensorDict):
        target = target[self.torque_guidance_obs_term]
      mean_action = self.actor(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0]
      )
      # Huber, not MSE: even after max_action_delta clamps the target and
      # the advantage mask drops fall-adjacent samples, remaining outliers
      # among the "good" states shouldn't dominate the batch average the
      # way a squared error lets them.
      loss = coef * F.smooth_l1_loss(mean_action[keep], target[keep])

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      self.optimizer.step()

      total_loss += loss.item()
      num_batches += 1

    return total_loss / max(num_batches, 1)
