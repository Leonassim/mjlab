"""Adversarial motion prior (AMP) on the RHPS1 tracking task (RHPS1_AMP=1).

The BWC copy walks like the BWC: constant pelvis height, flat feet, still arms.
Here the BWC stays only as the command schedule (directions, starts, stops,
standing segments, reset states). Its imitation terms are replaced by velocity
tracking of the direction command, and a discriminator trained on retargeted
human walking (scripts/tools/amp_dataset.py) rewards human-looking transitions
(Peng et al. 2021; Escontrela et al. 2022 for the LSGAN form used here).

The discriminator sees pairs (s_{t-k}, s_t), k = PAIR_STEPS policy steps, of
FEATURES computed identically in mjlab (env_features) and on the dataset
(offline_features). Observation, action and deployment format are unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

KEY_BODIES = ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK", "L_WRIST_Y_LINK", "R_WRIST_Y_LINK")
FEAT_DIM = 30 + 3 + 1 + 3 * len(KEY_BODIES) + 3 + 3
PAIR_STEPS = 6  # 30 ms at 200 Hz
VEL_TAU = 0.033  # base velocities low-passed like a 30 fps central difference


# ---------------------------------------------------------------------------
# Features: [joint pos (30, model order), projected gravity (3), base height (1),
# key bodies in the base frame (12), base lin vel (3), base ang vel (3)]
# ---------------------------------------------------------------------------


def env_features(env) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_apply_inverse

  robot = env.scene["robot"]
  d = robot.data
  if not hasattr(env, "_amp_body_ids"):
    names = list(robot.body_names)
    env._amp_body_ids = [names.index(n) for n in KEY_BODIES]
    env._amp_vel = torch.cat([d.root_link_lin_vel_b, d.root_link_ang_vel_b], -1).clone()
    env._amp_step = -1
  step = int(env.common_step_counter)
  if env._amp_step != step:
    v = torch.cat([d.root_link_lin_vel_b, d.root_link_ang_vel_b], -1)
    a = env.step_dt / VEL_TAU
    fresh = (env.episode_length_buf <= 1)[:, None]
    env._amp_vel = torch.where(fresh, v, env._amp_vel + a * (v - env._amp_vel))
    env._amp_step = step
  root = d.root_link_pos_w
  rel = d.body_link_pos_w[:, env._amp_body_ids] - root[:, None]
  q = d.root_link_quat_w[:, None].expand(-1, len(KEY_BODIES), -1)
  key = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape[0], -1)
  return torch.cat(
    [d.joint_pos, d.projected_gravity_b, root[:, 2:3], key, env._amp_vel], -1
  )


def offline_features(model, qpos: np.ndarray, lin_vel_w: np.ndarray, ang_vel_b: np.ndarray):
  """Same features from MuJoCo FK. qpos (T, 37) free base wxyz + 30 joints."""
  import mujoco

  d = mujoco.MjData(model)
  ids = [model.body(n).id for n in KEY_BODIES]
  out = np.zeros((len(qpos), FEAT_DIM), np.float32)
  for i, q in enumerate(qpos):
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    R = d.xmat[model.body("BODY").id].reshape(3, 3)
    root = d.xpos[model.body("BODY").id]
    key = ((d.xpos[ids] - root) @ R).reshape(-1)
    g = R.T @ np.array([0, 0, -1.0])
    out[i] = np.concatenate([q[7:], g, [root[2]], key, R.T @ lin_vel_w[i], ang_vel_b[i]])
  return out


# ---------------------------------------------------------------------------
# Velocity tracking of the direction command (replaces the BWC imitation)
# ---------------------------------------------------------------------------


def _target(env, command_name: str) -> torch.Tensor:
  from .env_cfgs import _filtered_velocity_command, _reference_velocity_command

  tau = float(os.environ.get("RHPS1_CMD_TAU", "0"))
  u = (_filtered_velocity_command(env, command_name, tau) if tau > 0
       else _reference_velocity_command(env, command_name))
  s = torch.tensor([float(x) for x in os.environ.get(
    "RHPS1_AMP_SPEED", "0.25,0.12,0.35").split(",")], device=u.device)
  return u * s


def track_lin_vel(env, command_name: str, std: float) -> torch.Tensor:
  v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  e = ((v - _target(env, command_name)[:, :2]) ** 2).sum(-1)
  return torch.exp(-e / std**2)


def track_yaw_rate(env, command_name: str, std: float) -> torch.Tensor:
  w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  return torch.exp(-((w - _target(env, command_name)[:, 2]) ** 2) / std**2)


# ---------------------------------------------------------------------------
# Discriminator and PPO
# ---------------------------------------------------------------------------


class _Normalizer(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    n = x.shape[0]
    m, v = x.mean(0), x.var(0, unbiased=False)
    tot = self.count + n
    delta = m - self.mean
    self.mean += delta * n / tot
    self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
    self.count = tot

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-6), -5.0, 5.0)


class Discriminator(nn.Module):
  def __init__(self, dim: int, hidden=(1024, 512)):
    super().__init__()
    layers, d = [], dim
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ReLU()]
      d = h
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(d, 1)
    self.norm = _Normalizer(dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(self.norm(x))).squeeze(-1)


def _mirror_qpos(qpos: np.ndarray, joint_names: list[str]) -> np.ndarray:
  from mjlab.tasks.velocity.config.rhps1.rl_ext import _joint_perm_sign

  perm, sign = _joint_perm_sign(joint_names)
  out = qpos.copy()
  out[:, 1] = -qpos[:, 1]
  out[:, 4] = -qpos[:, 4]  # quat wxyz: mirror about the xz plane flips x and z
  out[:, 6] = -qpos[:, 6]
  out[:, 7:] = qpos[:, 7:][:, perm] * np.array(sign)
  return out


def load_expert(path: str, device) -> tuple[torch.Tensor, torch.Tensor]:
  """Features (N, FEAT_DIM) and the indices i whose (i - PAIR_STEPS, i) pair is valid."""
  z = np.load(path)
  f = torch.tensor(z["features"], dtype=torch.float32, device=device)
  idx = torch.tensor(z["pair_end"], dtype=torch.long, device=device)
  return f, idx


from .rl_ext import RHPS1PPO  # noqa: E402


class RHPS1AMPPPO(RHPS1PPO):
  """PPO + AMP style reward. Env vars: RHPS1_AMP_DATA, RHPS1_AMP_W, RHPS1_AMP_LR,
  RHPS1_AMP_GP, RHPS1_AMP_BATCH."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    dev = self.device
    self.amp_w = float(os.environ.get("RHPS1_AMP_W", "2.0"))
    self.amp_dt = 0.005
    self.amp_gp = float(os.environ.get("RHPS1_AMP_GP", "5.0"))
    self.amp_batch = int(os.environ.get("RHPS1_AMP_BATCH", "4096"))
    self.expert, self.expert_idx = load_expert(os.environ["RHPS1_AMP_DATA"], dev)
    self.disc = Discriminator(2 * FEAT_DIM).to(dev)
    self.disc.norm.update(torch.cat([self.expert[self.expert_idx - PAIR_STEPS],
                                     self.expert[self.expert_idx]], -1))
    lr = float(os.environ.get("RHPS1_AMP_LR", "1e-4"))
    self.disc_opt = torch.optim.Adam([
      {"params": self.disc.trunk.parameters(), "weight_decay": 1e-4},
      {"params": self.disc.head.parameters(), "weight_decay": 1e-2},
    ], lr=lr)
    self._hist = None
    self._policy_pairs: list[torch.Tensor] = []
    self._style_sum = 0.0
    self._style_n = 0

  def _style(self, pair: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      d = self.disc(pair)
    return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)

  def process_env_step(self, obs, rewards, dones, extras):
    f = obs["amp"].to(self.device)
    if self._hist is None:
      self._hist = f[None].repeat(PAIR_STEPS + 1, 1, 1)
      self._age = torch.zeros(f.shape[0], dtype=torch.long, device=self.device)
    self._hist = torch.cat([self._hist[1:], f[None]], 0)
    self._age += 1
    valid = (self._age >= PAIR_STEPS) & ~dones.bool().to(self.device)
    pair = torch.cat([self._hist[0], self._hist[-1]], -1)
    r = self._style(pair) * valid
    self._style_sum += float(r.sum())
    self._style_n += int(valid.sum())
    if valid.any():
      self._policy_pairs.append(pair[valid].detach())
    d = dones.bool().to(self.device)
    if d.any():
      self._hist[:, d] = f[d][None]
      self._age[d] = 0
    super().process_env_step(obs, rewards + (self.amp_w * self.amp_dt * r).to(rewards.device),
                             dones, extras)

  def _disc_step(self, pol: torch.Tensor) -> dict[str, float]:
    n = min(self.amp_batch, pol.shape[0])
    p = pol[torch.randint(0, pol.shape[0], (n,), device=self.device)]
    i = self.expert_idx[torch.randint(0, len(self.expert_idx), (n,), device=self.device)]
    e = torch.cat([self.expert[i - PAIR_STEPS], self.expert[i]], -1)
    self.disc.norm.update(torch.cat([p, e], 0))
    e_n = self.disc.norm(e).detach().requires_grad_(True)
    d_e = self.disc.head(self.disc.trunk(e_n)).squeeze(-1)
    d_p = self.disc(p)
    loss = 0.5 * ((d_e - 1) ** 2).mean() + 0.5 * ((d_p + 1) ** 2).mean()
    (g,) = torch.autograd.grad(d_e.sum(), e_n, create_graph=True)
    gp = (g**2).sum(-1).mean()
    self.disc_opt.zero_grad()
    (loss + self.amp_gp * gp).backward()
    nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
    self.disc_opt.step()
    return {"loss": float(loss), "gp": float(gp), "d_e": float(d_e.mean()), "d_p": float(d_p.mean())}

  def update(self):
    loss = super().update()
    pol = torch.cat(self._policy_pairs, 0) if self._policy_pairs else None
    self._policy_pairs = []
    if pol is not None and pol.shape[0] > 0:
      acc: dict[str, float] = {}
      steps = self.num_learning_epochs * self.num_mini_batches
      for _ in range(steps):
        for k, v in self._disc_step(pol).items():
          acc[k] = acc.get(k, 0.0) + v / steps
      for k, v in acc.items():
        loss[f"amp_{k}"] = v
    if self._style_n:
      loss["amp_style"] = self._style_sum / self._style_n
      self._style_sum, self._style_n = 0.0, 0
    return loss

  def save(self) -> dict:
    out = super().save()
    out["amp_disc_state_dict"] = self.disc.state_dict()
    out["amp_disc_optimizer_state_dict"] = self.disc_opt.state_dict()
    return out

  def load(self, loaded_dict: dict, load_cfg, strict: bool):
    it = super().load(loaded_dict, load_cfg, strict)
    if "amp_disc_state_dict" in loaded_dict and (load_cfg is None or load_cfg.get("amp", True)):
      self.disc.load_state_dict(loaded_dict["amp_disc_state_dict"])
      self.disc_opt.load_state_dict(loaded_dict["amp_disc_optimizer_state_dict"])
    return it
