"""Depth-image student for the stairs: what the head Azure sees, with no map and no odometry.

Family of Agarwal 2022 / Extreme Parkour (Cheng 2023) / Humanoid Parkour (Zhuang 2024): a small depth image
from the head goes through a CNN, then a GRU with the proprioception, then an MLP. The GRU's memory keeps the
treads seen ahead that the head no longer sees under the feet. Trained by DAgger from the exact-scan teacher
(scripts/stairs/depth_distill.py).

RHPS1_DEPTH=1 adds, on top of the stairs recipe:
  - the `depth_cam` ray-cast sensor: a pinhole of DEPTH_W x DEPTH_H rays from the Azure pose of vision_map.py
    (CAM_POS / CAM_RPY on HEAD_P_LINK), the robot's own collision geoms occlude (group 3);
  - the `depth` observation group: the delayed, degraded image, flattened (H * W), in [0, 1] (range / max range,
    0 = no return, as the Azure reports invalid pixels);
  - the head held at HEAD_P (RHPS1_HEAD_HOLD), as on the robot while walking.
The teacher keeps reading the exact scan in the `actor` group; the student reads `actor` minus the scan + `depth`.

Degradations (training only, RHPS1_DEPTH_* to change), from the Azure Kinect and the stairs literature:
  - 25 Hz frames (PERIOD policy steps), 80 ms latency (LAT): the policy reads the last delivered frame;
  - range noise (NOISE, m), multiplicative (SCALE) and additive (BIAS, m) per-episode errors;
  - mount error per episode: image shifted by up to SHIFT pixels (about 1 deg per pixel at 64 px over 86 deg);
  - missing pixels (DROP), and mixed pixels at depth edges (EDGE_P): ToF pixels straddling a tread nosing
    return a blend of foreground and background, or nothing;
  - whole-frame blackouts (BLACK_P per frame, BLACK_T s): the policy must fall back on its memory.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from torch import nn

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.sensor.raycast_sensor import PinholeCameraPatternCfg


def _env(k: str, d: str) -> str:
  return os.environ.get("RHPS1_DEPTH_" + k, d)


DEPTH = os.environ.get("RHPS1_DEPTH", "0") == "1"
W = int(_env("W", "64"))
H = int(_env("H", "48"))
FOVY = math.radians(float(_env("FOVY", "70")))  # vertical; horizontal from the 4:3 aspect (~86 deg)
# The virtual image is tilted down inside the Azure's 120 deg WFOV circle (half-angle CONE): at 25 deg of head
# pitch the upper half of an untilted frame looks past the 2.88 m range. Pixels outside the circle never return.
PITCH = math.radians(float(_env("PITCH_DEG", "25")))
CONE = math.radians(float(_env("CONE_DEG", "60")))
RANGE = tuple(float(x) for x in _env("RANGE", "0.25,2.88").split(","))
PERIOD = int(_env("PERIOD", "2"))  # policy steps per frame (25 Hz at 50 Hz)
LAT = float(_env("LAT", "0.08"))  # s
NOISE = float(_env("NOISE", "0.01"))
SCALE = float(_env("SCALE", "0.01"))
BIAS = float(_env("BIAS", "0.01"))
SHIFT = int(_env("SHIFT", "1"))
DROP = float(_env("DROP", "0.05"))
EDGE = float(_env("EDGE", "0.04"))  # m, depth jump that makes an edge
EDGE_P = float(_env("EDGE_P", "0.5"))
BLACK_P = float(_env("BLACK_P", "0.002"))
BLACK_T = float(_env("BLACK_T", "0.5"))
# RANDLEVEL=1: each episode draws its own strength of every degradation, from none up to twice the value above
# (noise, drop; mixed-pixel probability up to 0.8), and its own latency (LAT +- LAT_J s). A student trained on one
# fixed level fell on clean images (depth2 1000: 25 arrivals / 8 falls degraded, 1 / 97 clean): the real Azure
# matches neither, so the student must hold over the whole range.
RANDLEVEL = _env("RANDLEVEL", "0") == "1"
LAT_J = float(_env("LAT_J", "0.04"))


@dataclass
class HeadDepthPatternCfg(PinholeCameraPatternCfg):
  """W x H pinhole rays in HEAD_P_LINK, from the Azure depth frame of vision_map.py (MuJoCo camera axes)."""

  def generate_rays(self, mj_model, device):
    from mjlab.tasks.tracking.config.rhps1.vision_map import CAM_POS, CAM_RPY, _rot_rpy

    ty = math.tan(FOVY / 2)
    tx = ty * W / H
    v = torch.linspace(ty, -ty, H)  # rows top to bottom
    u = torch.linspace(-tx, tx, W)
    gv, gu = torch.meshgrid(v, u, indexing="ij")
    d_img = torch.stack([gu, gv, -torch.ones_like(gu)], -1).reshape(-1, 3)
    d_img = d_img / d_img.norm(dim=-1, keepdim=True)
    c, sn = math.cos(PITCH), math.sin(PITCH)
    tilt = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, sn], [0.0, -sn, c]])  # about camera x: -z toward -y (down)
    d_cam = (tilt @ d_img.T).T
    dirs = (_rot_rpy(*CAM_RPY) @ d_cam.T).T
    offs = torch.tensor(CAM_POS).expand(dirs.shape[0], 3).clone()
    return offs.to(device), dirs.to(device)


def camera_sensor_cfg() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="depth_cam", frame=ObjRef(type="body", name="HEAD_P_LINK", entity="robot"),
    pattern=HeadDepthPatternCfg(), ray_alignment="base", max_distance=RANGE[1], include_geom_groups=(0, 3),
  )


class _DepthState:
  def __init__(self, env, train: bool):
    n, dev = env.num_envs, env.device
    self.train = train
    self.lat = max(0, round(LAT / env.step_dt))
    self.lat_j = round(LAT_J / env.step_dt) if RANDLEVEL else 0
    self.buf = torch.zeros(self.lat + self.lat_j + 1, n, H, W, device=dev)  # raw frames, newest at the end
    nb = self.buf.shape[0]
    self.lag = torch.full((n,), self.lat, dtype=torch.long, device=dev)  # frames back from the newest
    self.noise = torch.full((n,), NOISE, device=dev)
    self.drop = torch.full((n,), DROP, device=dev)
    self.edge_p = torch.full((n,), EDGE_P, device=dev)
    assert nb >= 1
    self.out = torch.zeros(n, H, W, device=dev)
    self.scale = torch.ones(n, device=dev)
    self.bias = torch.zeros(n, device=dev)
    self.shift = torch.zeros(n, 2, dtype=torch.long, device=dev)
    self.black = torch.zeros(n, device=dev)  # s of blackout left
    self.step = 0

  def reset(self, ids: torch.Tensor, raw: torch.Tensor) -> None:
    k = len(ids)
    if k == 0:
      return
    dev = raw.device
    if self.train:
      self.scale[ids] = 1 + (torch.rand(k, device=dev) * 2 - 1) * SCALE
      self.bias[ids] = (torch.rand(k, device=dev) * 2 - 1) * BIAS
      self.shift[ids] = torch.randint(-SHIFT, SHIFT + 1, (k, 2), device=dev)
      if RANDLEVEL:
        self.noise[ids] = torch.rand(k, device=dev) * 2 * NOISE
        self.drop[ids] = torch.rand(k, device=dev) * 2 * DROP
        self.edge_p[ids] = torch.rand(k, device=dev) * min(0.8, 2 * EDGE_P)
        self.lag[ids] = self.lat + torch.randint(-self.lat_j, self.lat_j + 1, (k,), device=dev)
    self.black[ids] = 0.0
    self.buf[:, ids] = raw[ids][None]
    self.out[ids] = self._degrade(raw[ids], ids)

  def _degrade(self, d: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """d [K, H, W] in m, 0 = no return."""
    if not self.train:
      return d
    k = d.shape[0]
    dev = d.device
    valid = d > 0
    sh = self.shift[ids]
    if SHIFT > 0:
      for s in sh.unique(dim=0):
        m = (sh == s).all(-1)
        d[m] = torch.roll(d[m], shifts=(int(s[0]), int(s[1])), dims=(1, 2))
        valid[m] = torch.roll(valid[m], shifts=(int(s[0]), int(s[1])), dims=(1, 2))
    d = d * self.scale[ids, None, None] + self.bias[ids, None, None] + torch.randn_like(d) * self.noise[ids, None, None]
    if EDGE_P > 0:
      # Mixed pixels: a pixel with a neighbour more than EDGE away (a nosing) returns a blend, or nothing.
      pad = torch.nn.functional.pad(d[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
      nb = torch.stack([pad[:, :-2, 1:-1], pad[:, 2:, 1:-1], pad[:, 1:-1, :-2], pad[:, 1:-1, 2:]], 0)
      jump = (nb - d[None]).abs()
      j, idx = jump.max(0)
      far = nb.gather(0, idx[None])[0]
      edge = (j > EDGE) & (torch.rand_like(d) < self.edge_p[ids, None, None])
      t = torch.rand_like(d)
      blend = d + t * (far - d)
      d = torch.where(edge, torch.where(torch.rand_like(d) < 0.3, torch.zeros_like(d), blend), d)
    drop = torch.rand_like(d) < self.drop[ids, None, None]
    d = torch.where(valid & ~drop & (d >= RANGE[0]) & (d <= RANGE[1]), d, torch.zeros_like(d))
    blk = self.black[ids] > 0
    return torch.where(blk[:, None, None], torch.zeros_like(d), d)

  def advance(self, env, raw: torch.Tensor) -> None:
    self.buf = torch.roll(self.buf, -1, 0)
    self.buf[-1] = raw
    self.step += 1
    if self.train and BLACK_P > 0:
      self.black = (self.black - env.step_dt).clamp(min=0)
      start = (torch.rand_like(self.black) < BLACK_P * PERIOD) & (self.black <= 0)
      self.black = torch.where(start, torch.full_like(self.black, BLACK_T), self.black)
    if self.step % PERIOD == 0:
      ids = torch.arange(env.num_envs, device=raw.device)
      nb = self.buf.shape[0]
      frame = self.buf[nb - 1 - self.lag, ids]  # per-env latency
      self.out = self._degrade(frame.clone(), ids)


_IN_CONE: dict = {}


def _cone_mask(device) -> torch.Tensor:
  m = _IN_CONE.get(device)
  if m is None:
    ty = math.tan(FOVY / 2)
    tx = ty * W / H
    gv, gu = torch.meshgrid(torch.linspace(ty, -ty, H), torch.linspace(-tx, tx, W), indexing="ij")
    d = torch.stack([gu, gv, -torch.ones_like(gu)], -1)
    d = d / d.norm(dim=-1, keepdim=True)
    c, sn = math.cos(PITCH), math.sin(PITCH)
    z = -(-sn * d[..., 1] + c * d[..., 2])  # component along the Azure axis (camera -z)
    m = _IN_CONE[device] = (z >= math.cos(CONE)).to(device)
  return m


def _raw(env) -> torch.Tensor:
  d = env.scene["depth_cam"].data.distances.view(env.num_envs, H, W)
  ok = (d >= RANGE[0]) & (d <= RANGE[1]) & _cone_mask(d.device)
  return torch.where(ok, d, torch.zeros_like(d))


def depth_image(env, train: bool = True) -> torch.Tensor:
  """Delayed, degraded depth frame, flattened, range / max range in [0, 1] (0 = no return)."""
  st = getattr(env, "_depth", None)
  raw = _raw(env)
  if st is None:
    st = env._depth = _DepthState(env, train=train and os.environ.get("RHPS1_DEPTH_CLEAN", "0") != "1")
    st.reset(torch.arange(env.num_envs, device=env.device), raw)
  else:
    fresh = (env.episode_length_buf == 0).nonzero().flatten()
    st.advance(env, raw)
    st.reset(fresh, raw)
  return (st.out / RANGE[1]).view(env.num_envs, -1)


def apply(cfg, play: bool = False) -> None:
  cfg.scene.sensors = cfg.scene.sensors + (camera_sensor_cfg(),)
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={"image": ObservationTermCfg(func=depth_image, params={"train": not play})}, concatenate_terms=True, enable_corruption=False)


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------


class DepthStudent(nn.Module):
  """depth [B, H*W] -> CNN -> 64; [proprio (normalized), cnn] -> GRU(256) -> MLP -> actions.

  The proprio normalizer is the teacher's (frozen slice), so both read the same scaled inputs.
  """

  def __init__(self, n_prop: int, n_act: int, mean: torch.Tensor, std: torch.Tensor, gru: int = 256,
               cnn_out: int = 64) -> None:
    super().__init__()
    self.n_prop, self.h, self.w = n_prop, H, W
    self.register_buffer("mean", mean.clone())
    self.register_buffer("std", std.clone())
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
      nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
      nn.Flatten(),
    )
    with torch.no_grad():
      nf = self.cnn(torch.zeros(1, 1, H, W)).shape[-1]
    self.cnn_fc = nn.Sequential(nn.Linear(nf, cnn_out), nn.ELU())
    self.gru = nn.GRU(n_prop + cnn_out, gru, batch_first=False)
    self.mlp = nn.Sequential(nn.Linear(gru + n_prop, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                             nn.Linear(128, n_act))
    self.gru_dim = gru

  def step(self, prop: torch.Tensor, depth: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One step: prop [B, P], depth [B, H*W], h [1, B, G] -> actions [B, A], h."""
    a, h = self.seq(prop[None], depth[None], h)
    return a[0], h

  def seq(self, prop: torch.Tensor, depth: torch.Tensor, h: torch.Tensor,
          dones: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """prop [T, B, P], depth [T, B, H*W], dones [T, B] (reset AFTER step t) -> actions [T, B, A], h."""
    t, b = prop.shape[:2]
    x = ((prop - self.mean) / self.std).clamp(-5, 5)
    z = self.cnn_fc(self.cnn(depth.reshape(t * b, 1, self.h, self.w))).view(t, b, -1)
    inp = torch.cat([x, z], -1)
    if dones is None:
      y, h = self.gru(inp, h)
    else:
      ys = []
      for i in range(t):
        yi, h = self.gru(inp[i : i + 1], h)
        ys.append(yi)
        h = h * (1 - dones[i].float())[None, :, None]
      y = torch.cat(ys, 0)
    return self.mlp(torch.cat([y, x], -1)), h


class DepthStudentOnnx(nn.Module):
  """Export: (proprio [1, P], depth [1, H*W], h [1, 1, G]) -> (actions [1, A], h_out)."""

  def __init__(self, m: DepthStudent) -> None:
    super().__init__()
    self.m = m

  def forward(self, prop, depth, h):
    return self.m.step(prop, depth, h)
