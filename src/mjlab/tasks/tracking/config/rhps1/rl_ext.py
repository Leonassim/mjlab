"""Mirror loss for the RHPS1 tracking task (rsl-rl Symmetry extension).

Every tracking checkpoint limps: one swing lasts 10-40 % longer than the other,
and which side flips between checkpoints (J: 0.62/0.85 s at 5000, 0.74/0.64 at
15000, 0.84/0.57 at 17500) while the BWC is symmetric (0.61/0.61). The mirror
loss asks the actor to map mirrored observations to mirrored actions, as the
velocity task already does.

Only groups whose every term has a rule are mirrored. The tracking critic
carries privileged terms (the motion `command`, anchor pose, body positions)
that the velocity rules would mirror wrongly; it is left unchanged, which is
enough for the mirror loss (it only evaluates the actor). Keep
use_data_augmentation False.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.tasks.velocity.config.rhps1.rl_ext import (
  _VEC3_ANG_SIGN,
  _VEC3_CMD_SIGN,
  _VEC3_LIN_SIGN,
  _joint_perm_sign,
  _tile_block,
)

_RULES = {
  "base_lin_vel": ([0, 1, 2], list(_VEC3_LIN_SIGN)),
  "projected_gravity": ([0, 1, 2], list(_VEC3_LIN_SIGN)),
  "base_ang_vel": ([0, 1, 2], list(_VEC3_ANG_SIGN)),
  "velocity_command": ([0, 1, 2], list(_VEC3_CMD_SIGN)),
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")

_specs: dict[int, dict] = {}


def _build(env) -> dict:
  u = env.unwrapped
  device = u.device
  jperm, jsign = _joint_perm_sign(list(u.scene["robot"].joint_names))
  out: dict = {"groups": {}}
  om = u.observation_manager
  for group, names in om.active_terms.items():
    if not all(n in _RULES or n in _JOINT_TERMS for n in names):
      continue
    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for name, dims in zip(names, om.group_obs_term_dim[group]):
      total = int(np.prod(dims))
      bp, bs = (jperm, jsign) if name in _JOINT_TERMS else _RULES[name]
      p, s = _tile_block(bp, bs, total, offset)
      perm += p
      sign += s
      offset += total
    out["groups"][group] = (
      torch.tensor(perm, dtype=torch.long, device=device),
      torch.tensor(sign, device=device),
    )
  am = u.action_manager
  perm, sign, offset = [], [], 0
  for term_name in am.active_terms:
    term = am.get_term(term_name)
    bp, bs = _joint_perm_sign(list(term.target_names))
    p, s = _tile_block(bp, bs, term.action_dim, offset)
    perm += p
    sign += s
    offset += term.action_dim
  out["actions"] = (
    torch.tensor(perm, dtype=torch.long, device=device),
    torch.tensor(sign, device=device),
  )
  return out


def rhps1_tracking_mirror(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
  """Returns [original; mirrored] along the batch dimension, as rsl-rl expects."""
  if id(env) not in _specs:
    _specs[id(env)] = _build(env)
  sp = _specs[id(env)]
  obs_out = None
  if obs is not None:
    mirrored = {}
    for key in obs.keys():
      if key in sp["groups"]:
        perm, sign = sp["groups"][key]
        mirrored[key] = obs[key][..., perm] * sign
      else:
        mirrored[key] = obs[key]
    obs_out = torch.cat([obs, TensorDict(mirrored, batch_size=obs.batch_size)], dim=0)
  actions_out = None
  if actions is not None:
    perm, sign = sp["actions"]
    actions_out = torch.cat([actions, actions[..., perm] * sign], dim=0)
  return obs_out, actions_out


# ---------------------------------------------------------------------------
# Lipschitz gradient penalty (RHPS1_LIPSCHITZ_W), after Chen et al., "Learning
# Smooth Humanoid Locomotion through Lipschitz-Constrained Policies".
#
# The real robot holds a 5.5 Hz limit cycle at rest after walking (2026-10-15):
# the drives follow their commands normally, the body twists ~0.5 deg in yaw
# (8x what the hip yaw encoders see, a compliance the sim does not have) and the
# policy, reacting to the gyro and the estimated velocity, keeps it going.
# Bounding how much the actions move per unit of observation lowers that loop
# gain whatever the unmodelled mode is.
#
# The gradient is taken w.r.t. the NORMALISED actor input, so one input std
# weighs the same in every unit, and only on the inputs that close the loop
# (RHPS1_LIPSCHITZ_OBS, actor-obs index ranges; default: base lin vel history,
# base ang vel, joint pos, joint vel -- not gravity, command or last action).
# ||J||_F^2 is estimated with one random projection per sample (Hutchinson).
# ---------------------------------------------------------------------------

import os as _os

from rsl_rl.algorithms import PPO as _PPO


class _MirrorAndLipschitz:
  """Stands in for rsl-rl's Symmetry object: mirror loss (if any) + gradient penalty."""

  use_data_augmentation = False
  use_mirror_loss = True
  mirror_loss_coeff = 1.0

  def __init__(self, inner, coeff: float, ranges: list[tuple[int, int]], samples: int):
    self.inner = inner
    self.coeff = coeff
    self.ranges = ranges
    self.samples = samples
    self.use_data_augmentation = bool(inner and inner.use_data_augmentation)
    self.gp_sum = 0.0
    self.gp_n = 0

  def augment_batch(self, batch, original_batch_size):
    if self.inner:
      self.inner.augment_batch(batch, original_batch_size)

  def compute_loss(self, actor, batch, original_batch_size):
    loss = torch.zeros((), device=batch.actions.device)
    if self.inner:
      m = self.inner.compute_loss(actor, batch, original_batch_size)
      if self.inner.use_mirror_loss:
        loss = loss + self.inner.mirror_loss_coeff * m
    n = min(original_batch_size, self.samples)
    obs = batch.observations[:n]
    group = actor.obs_groups[0]
    assert len(actor.obs_groups) == 1, "gradient penalty expects a single actor obs group"
    x = obs[group].detach().clone().requires_grad_(True)
    td = obs.clone()
    td[group] = x
    mu = actor(td)
    v = torch.randn_like(mu)
    (g,) = torch.autograd.grad((mu * v).sum(), x, create_graph=True)
    norm = actor.obs_normalizer
    if hasattr(norm, "_std"):
      g = g * (norm._std + norm.eps)
    mask = torch.zeros(g.shape[-1], dtype=torch.bool, device=g.device)
    for a, b in self.ranges:
      mask[a:b] = True
    gp = (g[:, mask] ** 2).sum(-1).mean()
    self.gp_sum += float(gp.detach())
    self.gp_n += 1
    return loss + self.coeff * gp


class RHPS1PPO(_PPO):
  """PPO with the optional Lipschitz gradient penalty above."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    coeff = float(_os.environ.get("RHPS1_LIPSCHITZ_W", "0"))
    if coeff > 0:
      spec = _os.environ.get("RHPS1_LIPSCHITZ_OBS", "0:18,21:81")
      ranges = [tuple(int(v) for v in r.split(":")) for r in spec.split(",")]
      samples = int(_os.environ.get("RHPS1_LIPSCHITZ_SAMPLES", "4096"))
      self.symmetry = _MirrorAndLipschitz(self.symmetry, coeff, ranges, samples)

  def update(self):
    loss = super().update()
    if isinstance(self.symmetry, _MirrorAndLipschitz) and self.symmetry.gp_n:
      loss["lipschitz_gp"] = self.symmetry.gp_sum / self.symmetry.gp_n
      self.symmetry.gp_sum, self.symmetry.gp_n = 0.0, 0
    return loss
