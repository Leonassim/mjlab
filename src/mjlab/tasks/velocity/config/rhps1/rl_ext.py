"""RHPS1-specific rsl-rl extensions: held exploration noise and mirror symmetry.

Everything here plugs into stock rsl-rl through its string-resolved config
hooks (``distribution_cfg.class_name`` and ``symmetry_cfg.data_augmentation_func``),
so no rsl-rl fork is needed and upstream mjlab files are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

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
}
_JOINT_SPACE_TERMS = ("joint_pos", "joint_vel", "actions")

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
