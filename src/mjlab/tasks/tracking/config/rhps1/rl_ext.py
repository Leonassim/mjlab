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
