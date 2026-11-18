"""Onboard height map for the stairs student: what the head-mounted Azure depth camera can actually build.

The teacher's scan (stairs_scan) is exact terrain under a 25 x 13 grid. Here the same grid is read from a map
built like a real elevation map: depth rays from the camera pose of the URDF (azure_depth_frame on HEAD_P_LINK),
range and cone limits, occlusion by the terrain itself, a per-episode extrinsic error, range noise, latency,
a rolling memory in a drifting odometry frame, and footprints written under loaded feet. Same 325 values and
convention (pelvis z - terrain z - offset), so a teacher checkpoint loads as is.

Map: world-aligned (odometry axes) torus of MAP_N x MAP_N cells, each slot tagged with the absolute cell it holds;
a slot whose tag differs is unknown. Unknown cells read as the lowest loaded sole (flat at the support level).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import torch

from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.sensor.raycast_sensor import PinholeCameraPatternCfg


def _env(k: str, d: str) -> str:
  return os.environ.get("RHPS1_VIS_" + k, d)


VIS = os.environ.get("RHPS1_VIS", "0") == "1"
# Head held at HEAD_P without the onboard map (depth-image student, depth_student.py). VIS implies it.
HEAD_HOLD = VIS or os.environ.get("RHPS1_HEAD_HOLD", "0") == "1"
HEAD_P = math.radians(float(_env("HEAD_P_DEG", "25")))  # head pitch held during walking (Leo, 2026-11-17)
# azure_depth_frame in HEAD_P_LINK, MuJoCo camera axes (-z forward). The default is Choreonoid's (the ZED Mini
# mount); the real Azure, fitted on rhps1 at three head angles (2026-11-18): CAM_POS 0.1190,0.0017,0.1825 and
# CAM_RPY 1.49513,-0.00503,-1.57095.
CAM_POS = tuple(float(x) for x in _env("CAM_POS", "0.071,0.0017,0.1558").split(","))
CAM_RPY = tuple(float(x) for x in _env("CAM_RPY", "1.46608,1.45688e-06,-1.5708").split(","))
CONE = math.radians(float(_env("CONE_DEG", "60")))  # half-angle; WFOV is a 120 deg circle
RANGE = tuple(float(x) for x in _env("RANGE", "0.25,2.88").split(","))  # WFOV 2x2 binned
ROI_X = tuple(float(x) for x in _env("ROI_X", "0.0,1.6").split(","))  # ground aimed at, ahead of the pelvis
ROI_Y = float(_env("ROI_Y", "0.6"))
ROI_RES = tuple(float(x) for x in _env("ROI_RES", "0.025,0.05").split(","))
PERIOD = int(_env("PERIOD", "2"))  # policy steps per depth frame (25 Hz at 50 Hz)
LAT = float(_env("LAT", "0.08"))  # s, frame to map
NOISE = float(_env("NOISE", "0.01"))  # m, along each ray
ROT_BIAS = math.radians(float(_env("ROT_BIAS_DEG", "1.0")))  # per-episode extrinsic, each axis
ROT_JIT = math.radians(float(_env("ROT_JIT_DEG", "0.3")))  # per frame
POS_BIAS = float(_env("POS_BIAS", "0.01"))
ODO_SCALE = float(_env("ODO_SCALE", "0.05"))  # per-episode horizontal distance error
ODO_ZSCALE = float(_env("ODO_ZSCALE", "0.1"))
ODO_RW = float(_env("ODO_RW", "0.003"))  # m / sqrt(s)
ODO_ZRW = float(_env("ODO_ZRW", "0.002"))
ODO_YAW = float(_env("ODO_YAW", "0.005"))  # rad / sqrt(s)
ALPHA = float(_env("ALPHA", "0.5"))  # weight of a new frame on a known cell
MAP_N = int(_env("MAP_N", "48"))
PREFILL = float(_env("PREFILL", "1.0"))  # share of episodes starting with the scan area mapped (walked in)
# Real stream and failures (all off by default). Measured on rhps1 (2026-11-18): 15 Hz, 45-65 ms median latency and
# up to 150 ms, 2.6-6 % gaps over 0.2 s; the controller reads a map older than vision_max_age as flat ground.
LAT_RANGE = tuple(float(x) for x in _env("LAT_RANGE", "").split(",") if x)  # s, per episode, replaces LAT
DROP = float(_env("DROP", "0"))  # per-frame loss probability
OUT_P = float(_env("OUT_P", "0"))  # outages started per second (camera or network down)
OUT_S = tuple(float(x) for x in _env("OUT_S", "0.2,3.0").split(","))
MAX_AGE = float(_env("MAX_AGE", "0"))  # s; > 0: the whole scan reads flat once the last frame is older
RAY_DROP = float(_env("RAY_DROP", "0"))  # per frame, up to this share of the rays return nothing
PHANTOM_P = float(_env("PHANTOM_P", "0"))  # phantom blocks started per second (people, reflections, clutter)
PHANTOM_H = tuple(float(x) for x in _env("PHANTOM_H", "0.15,0.6").split(","))  # m above the lowest sole
PHANTOM_S = tuple(float(x) for x in _env("PHANTOM_S", "0.5,4.0").split(","))  # s, rewritten every frame meanwhile
LOAD_N = 10.0
SOLE_DEPTH = 0.097  # ankle to sole bottom
FOOT_PTS = [(x, y) for x in (-0.08, -0.04, 0.0, 0.04, 0.08) for y in (-0.04, 0.0, 0.04)]
_OFF = 1 << 20


def _rot_rpy(r: float, p: float, y: float) -> torch.Tensor:
  def rx(a):
    return torch.tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

  def ry(a):
    return torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

  def rz(a):
    return torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

  return (rz(y) @ ry(p) @ rx(r)).float()


def _find(names, suffix: str) -> int:
  for i, n in enumerate(names):
    if n == suffix or n.endswith("/" + suffix):
      return i
  raise KeyError(suffix)


@dataclass
class DepthCamPatternCfg(PinholeCameraPatternCfg):
  """Pixels of the depth camera that see the ground ROI at the nominal stance (head at HEAD_P).

  The real image is ~1 cm per pixel there; only the pixels landing near map cells are kept. Rays are fixed
  in HEAD_P_LINK, so sway, head motion and terrain move their hits like real pixels."""

  def generate_rays(self, mj_model, device):
    d = mujoco.MjData(mj_model)
    head_j = _find([mj_model.joint(i).name for i in range(mj_model.njnt)], "HEAD_P")
    d.qpos[mj_model.jnt_qposadr[head_j]] = HEAD_P
    mujoco.mj_kinematics(mj_model, d)
    bn = [mj_model.body(i).name for i in range(mj_model.nbody)]
    hb, pb = _find(bn, "HEAD_P_LINK"), _find(bn, "BODY")
    r_head = torch.tensor(d.xmat[hb].reshape(3, 3)).float()
    r_pel = torch.tensor(d.xmat[pb].reshape(3, 3)).float()
    c_loc = torch.tensor(CAM_POS)
    cam = r_pel.T @ (torch.tensor(d.xpos[hb]).float() + r_head @ c_loc - torch.tensor(d.xpos[pb]).float())
    from mjlab.tasks.tracking.config.rhps1.stairs import SCAN_OFFSET

    xs = torch.arange(ROI_X[0], ROI_X[1] + 1e-6, ROI_RES[0])
    ys = torch.arange(-ROI_Y, ROI_Y + 1e-6, ROI_RES[1])
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    g = torch.stack([gx.flatten(), gy.flatten(), torch.full_like(gx.flatten(), -SCAN_OFFSET)], -1)
    dirs_pel = g - cam
    dirs = (r_pel.T @ r_head).T @ dirs_pel.T  # pelvis frame -> HEAD_P_LINK frame
    dirs = (dirs / dirs.norm(dim=0, keepdim=True)).T
    axis = _rot_rpy(*CAM_RPY) @ torch.tensor([0.0, 0.0, -1.0])
    keep = (dirs @ axis) >= math.cos(CONE)
    dirs = dirs[keep]
    offs = c_loc.expand(dirs.shape[0], 3).clone()
    return offs.to(device), dirs.to(device)


def camera_sensor_cfg() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="stairs_cam", frame=ObjRef(type="body", name="HEAD_P_LINK", entity="robot"),
    pattern=DepthCamPatternCfg(), ray_alignment="base", max_distance=RANGE[1], include_geom_groups=(0, 3),
  )  # group 3 = robot collision geoms: the body hides the ground, and its own points are filtered out


def _yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  w, xyz = q[..., :1], q[..., 1:]
  t = 2 * torch.cross(xyz.expand_as(v), v, dim=-1)
  return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


class VisionMap:
  def __init__(self, env):
    b, dev, n = env.num_envs, env.device, MAP_N
    self.n, self.res = n, None
    self.h = torch.zeros(b, n * n, device=dev)
    self.key = torch.full((b, n * n), -1, dtype=torch.long, device=dev)
    self.est = torch.zeros(b, 3, device=dev)
    self.prev = torch.zeros(b, 3, device=dev)
    self.yaw_err = torch.zeros(b, device=dev)
    self.s_xy = torch.zeros(b, device=dev)
    self.s_z = torch.zeros(b, device=dev)
    self.rot_bias = torch.zeros(b, 3, device=dev)
    self.pos_bias = torch.zeros(b, 3, device=dev)
    self.fifo: list[tuple[torch.Tensor, torch.Tensor]] = []
    frame_dt = env.step_dt * PERIOD
    self.frame_dt = frame_dt
    self.lat_n = max(0, round(LAT / frame_dt))
    self.lat_rng = (max(0, round(LAT_RANGE[0] / frame_dt)), max(0, round(LAT_RANGE[1] / frame_dt))) if LAT_RANGE else None
    self.lat_env = torch.full((b,), self.lat_n, dtype=torch.long, device=dev)
    self.out_until = torch.zeros(b, dtype=torch.long, device=dev)  # step counter
    self.last_frame = torch.full((b,), -(10**9), dtype=torch.long, device=dev)
    self.ph_until = torch.zeros(b, dtype=torch.long, device=dev)
    self.ph_box = torch.zeros(b, 5, device=dev)  # centre x y (odom), half x y, top z
    self.ph_yaw = torch.zeros(b, device=dev)
    g = torch.linspace(-1.0, 1.0, 9, device=dev)
    self.ph_grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).view(-1, 2)
    self.t = -1
    robot = env.scene["robot"]
    self.feet = [list(robot.body_names).index(k) for k in ("L_ANKLE_P_LINK", "R_ANKLE_P_LINK")]
    self.foot_pts = torch.tensor(FOOT_PTS, device=dev)
    from mjlab.tasks.tracking.config.rhps1.stairs import SCAN_OFFSET, SCAN_RES

    self.res, self.offset = SCAN_RES, SCAN_OFFSET
    self.scan_xy = env.scene["stairs_scan"]._local_offsets[:, :2].clone()
    cam = env.scene["stairs_cam"]
    self.robot_geom = torch.tensor(cam._mj_model.geom_group == 3, device=dev)

  # odometry frame: rigid map of the true world anchored at the pelvis, p_o = R(yaw_err)(p - pel) + est
  def _to_odom(self, p: torch.Tensor, pel: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(self.yaw_err), torch.sin(self.yaw_err)
    shp = (-1,) + (1,) * (p.dim() - 2)
    c, s = c.view(shp), s.view(shp)
    d = p - pel.view((pel.shape[0],) + (1,) * (p.dim() - 2) + (3,))
    x = c * d[..., 0] - s * d[..., 1]
    y = s * d[..., 0] + c * d[..., 1]
    e = self.est.view((pel.shape[0],) + (1,) * (p.dim() - 2) + (3,))
    return torch.stack([x + e[..., 0], y + e[..., 1], d[..., 2] + e[..., 2]], -1)

  def _cells(self, xy: torch.Tensor):
    ix = torch.floor(xy[..., 0] / self.res).long()
    iy = torch.floor(xy[..., 1] / self.res).long()
    return ix, iy

  def _slot_key(self, ix, iy):
    n = self.n
    return torch.remainder(ix, n) * n + torch.remainder(iy, n), (ix + _OFF) * (2 * _OFF) + (iy + _OFF)

  def _write(self, pts: torch.Tensor, valid: torch.Tensor, replace: bool) -> None:
    n2 = self.n * self.n
    ix, iy = self._cells(pts)
    cx, cy = self._cells(self.est[:, None, :2])
    half = self.n // 2
    valid = valid & ((ix - cx).abs() < half) & ((iy - cy).abs() < half)
    slot, key = self._slot_key(ix, iy)
    slot = torch.where(valid, slot, n2)
    w = valid.float()
    b = pts.shape[0]
    sm = torch.zeros(b, n2 + 1, device=pts.device).scatter_add_(1, slot, pts[..., 2] * w)[:, :n2]
    cnt = torch.zeros(b, n2 + 1, device=pts.device).scatter_add_(1, slot, w)[:, :n2]
    kb = torch.full((b, n2 + 1), -1, dtype=torch.long, device=pts.device).scatter_(1, slot, key)[:, :n2]
    hit = cnt > 0
    m = sm / cnt.clamp(min=1.0)
    if replace:
      self.h = torch.where(hit, m, self.h)
    else:
      same = hit & (self.key == kb)
      self.h = torch.where(same, (1 - ALPHA) * self.h + ALPHA * m, torch.where(hit, m, self.h))
    self.key = torch.where(hit, kb, self.key)

  def _reset(self, env, ids: torch.Tensor, pel: torch.Tensor) -> None:
    k = ids.numel()
    dev = pel.device
    self.prev[ids] = pel[ids]
    self.est[ids] = pel[ids]
    self.yaw_err[ids] = 0.0
    self.s_xy[ids] = (torch.rand(k, device=dev) * 2 - 1) * ODO_SCALE
    self.s_z[ids] = (torch.rand(k, device=dev) * 2 - 1) * ODO_ZSCALE
    self.rot_bias[ids] = (torch.rand(k, 3, device=dev) * 2 - 1) * ROT_BIAS
    self.pos_bias[ids] = (torch.rand(k, 3, device=dev) * 2 - 1) * POS_BIAS
    self.key[ids] = -1
    for _, v in self.fifo:
      v[ids] = False
    if self.lat_rng is not None:
      self.lat_env[ids] = torch.randint(self.lat_rng[0], self.lat_rng[1] + 1, (k,), device=dev)
    self.out_until[ids] = 0
    self.ph_until[ids] = 0
    self.last_frame[ids] = -(10**9)  # no map yet: flat under MAX_AGE, as the controller before the first map
    pre = ids[torch.rand(k, device=dev) < PREFILL]
    if pre.numel():
      self.last_frame[pre] = int(env.common_step_counter)
      hp = env.scene["stairs_scan"].data.hit_pos_w[pre]
      ok = env.scene["stairs_scan"].data.distances[pre] >= 0
      hp = hp.clone()
      hp[..., 2] += torch.randn_like(hp[..., 2]) * NOISE
      sub = _Sub(self, pre)
      sub.write(hp, ok)

  def _capture(self, env, pel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = env.scene["stairs_cam"].data
    q = d.frame_quat_w[:, 0]
    c = d.pos_w + _qrot(q, torch.tensor(CAM_POS, device=q.device).expand_as(d.pos_w))
    v = d.hit_pos_w - c[:, None]
    r = d.distances
    import warp as wp

    gid = wp.to_torch(env.scene["stairs_cam"]._ray_geomid).long()
    valid = (r >= RANGE[0]) & (r <= RANGE[1]) & (gid >= 0) & ~self.robot_geom[gid.clamp(min=0)]
    if RAY_DROP > 0:
      frac = torch.rand(r.shape[0], 1, device=r.device) * RAY_DROP
      valid &= torch.rand_like(r) >= frac
    u = v / r.clamp(min=1e-3)[..., None]
    rn = r + torch.randn_like(r) * NOISE
    w = self.rot_bias + torch.randn_like(self.rot_bias) * ROT_JIT  # camera axes
    w = _qrot(q, w)[:, None].expand_as(u)
    ray = u * rn[..., None]
    p = c[:, None] + ray + torch.cross(w, ray, dim=-1) + self.pos_bias[:, None]
    return self._to_odom(p, pel), valid

  def _footprints(self, env, pel: torch.Tensor) -> None:
    robot = env.scene["robot"]
    fp = robot.data.body_link_pos_w[:, self.feet]
    fy = _yaw(robot.data.body_link_quat_w[:, self.feet])
    from mjlab.tasks.tracking.config.rhps1.stairs import _feet

    _, _, _, _, fn = _feet(env)
    loaded = fn.view(fp.shape[0], 2, -1).sum(-1) > LOAD_N
    c, s = torch.cos(fy)[..., None], torch.sin(fy)[..., None]
    ox, oy = self.foot_pts[:, 0], self.foot_pts[:, 1]
    x = fp[..., 0:1] + c * ox - s * oy
    y = fp[..., 1:2] + s * ox + c * oy
    z = (fp[..., 2:3] - SOLE_DEPTH).expand_as(x)
    p = torch.stack([x, y, z], -1).flatten(1, 2)
    self._write(self._to_odom(p, pel), loaded[..., None].expand_as(x).flatten(1), replace=True)

  def step(self, env) -> None:
    t = int(env.common_step_counter)
    if t == self.t:
      return
    self.t = t
    robot = env.scene["robot"]
    pel = robot.data.root_link_pos_w
    reset = env.episode_length_buf == 0
    dpos = pel - self.prev
    dpos[reset] = 0.0
    self.prev = pel.clone()
    dt = env.step_dt
    b = pel.shape[0]
    self.yaw_err += torch.randn(b, device=pel.device) * ODO_YAW * math.sqrt(dt)
    c, s = torch.cos(self.yaw_err), torch.sin(self.yaw_err)
    dx, dy = (1 + self.s_xy) * dpos[:, 0], (1 + self.s_xy) * dpos[:, 1]
    rw = torch.randn(b, 3, device=pel.device) * math.sqrt(dt)
    self.est[:, 0] += c * dx - s * dy + rw[:, 0] * ODO_RW
    self.est[:, 1] += s * dx + c * dy + rw[:, 1] * ODO_RW
    self.est[:, 2] += (1 + self.s_z) * dpos[:, 2] + rw[:, 2] * ODO_ZRW
    if bool(reset.any()):
      self._reset(env, reset.nonzero().squeeze(-1), pel)
    self._footprints(env, pel)
    if t % PERIOD == 0:
      p, v = self._capture(env, pel)
      if DROP > 0 or OUT_P > 0:
        lost = torch.rand(b, device=pel.device) < DROP
        start = (torch.rand(b, device=pel.device) < OUT_P * self.frame_dt) & (self.out_until <= t)
        dur = OUT_S[0] + torch.rand(b, device=pel.device) * (OUT_S[1] - OUT_S[0])
        self.out_until = torch.where(start, t + (dur / dt).long(), self.out_until)
        v = v & ~(lost | (self.out_until > t))[:, None]
      self.fifo.append((p, v))
      if self.lat_rng is None:
        while len(self.fifo) > self.lat_n:
          p, v = self.fifo.pop(0)
          self._write(p, v, replace=False)
          self.last_frame = torch.where(v.any(-1), t, self.last_frame)
      else:  # per-episode latency: each env writes the frame whose age matches its own
        n = len(self.fifo)
        for k, (p, v) in enumerate(self.fifo):
          due = (self.lat_env == n - 1 - k)[:, None] & v
          if bool(due.any()):
            self._write(p, due, replace=False)
            self.last_frame = torch.where(due.any(-1), t, self.last_frame)
        while len(self.fifo) > self.lat_rng[1]:  # older than the longest latency: written by every env
          self.fifo.pop(0)
      if PHANTOM_P > 0:
        self._phantoms(env, pel, t, dt)

  def _phantoms(self, env, pel: torch.Tensor, t: int, dt: float) -> None:
    """Blocks that are not there, ahead of the robot, rewritten every frame while they last."""
    b, dev = pel.shape[0], pel.device
    start = (torch.rand(b, device=dev) < PHANTOM_P * self.frame_dt) & (self.ph_until <= t)
    if bool(start.any()):
      ids = start.nonzero().squeeze(-1)
      k = ids.numel()
      robot = env.scene["robot"]
      yaw = _yaw(robot.data.root_link_quat_w[ids]) + self.yaw_err[ids]
      fwd = 0.3 + torch.rand(k, device=dev) * 1.0
      lat = (torch.rand(k, device=dev) * 2 - 1) * 0.5
      c, s = torch.cos(yaw), torch.sin(yaw)
      top = self._fill(env)[ids] + PHANTOM_H[0] + torch.rand(k, device=dev) * (PHANTOM_H[1] - PHANTOM_H[0])
      half = 0.075 + torch.rand(k, 2, device=dev) * 0.175
      self.ph_box[ids] = torch.stack(
        [self.est[ids, 0] + c * fwd - s * lat, self.est[ids, 1] + s * fwd + c * lat, half[:, 0], half[:, 1], top], -1)
      self.ph_yaw[ids] = yaw
      dur = PHANTOM_S[0] + torch.rand(k, device=dev) * (PHANTOM_S[1] - PHANTOM_S[0])
      self.ph_until[ids] = t + (dur / dt).long()
    act = self.ph_until > t
    if bool(act.any()):
      bx = self.ph_box
      c, s = torch.cos(self.ph_yaw)[:, None], torch.sin(self.ph_yaw)[:, None]
      gx, gy = self.ph_grid[None, :, 0] * bx[:, 2:3], self.ph_grid[None, :, 1] * bx[:, 3:4]
      pts = torch.stack([bx[:, :1] + c * gx - s * gy, bx[:, 1:2] + s * gx + c * gy, bx[:, 4:5].expand_as(gx)], -1)
      self._write(pts, act[:, None].expand(-1, pts.shape[1]), replace=True)

  def read(self, env) -> torch.Tensor:
    robot = env.scene["robot"]
    yaw = _yaw(robot.data.root_link_quat_w) + self.yaw_err
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    ox, oy = self.scan_xy[:, 0], self.scan_xy[:, 1]
    px = self.est[:, :1] + c * ox - s * oy
    py = self.est[:, 1:2] + s * ox + c * oy
    fx, fy = px / self.res - 0.5, py / self.res - 0.5
    i0, j0 = torch.floor(fx), torch.floor(fy)
    wx, wy = fx - i0, fy - j0
    i0, j0 = i0.long(), j0.long()
    fill = self._fill(env)
    out = torch.zeros_like(px)
    for di, dj, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
      slot, key = self._slot_key(i0 + di, j0 + dj)
      known = self.key.gather(1, slot) == key
      out += w * torch.where(known, self.h.gather(1, slot), fill[:, None])
    if MAX_AGE > 0:  # the controller's vision_max_age: a stale or missing map reads as flat at the lowest sole
      stale = (int(env.common_step_counter) - self.last_frame) * env.step_dt > MAX_AGE
      out = torch.where(stale[:, None], fill[:, None].expand_as(out), out)
    return (self.est[:, 2:3] - out - self.offset).clamp(-1.0, 1.0)

  def _fill(self, env) -> torch.Tensor:
    robot = env.scene["robot"]
    pel = robot.data.root_link_pos_w
    sole = robot.data.body_link_pos_w[:, self.feet, 2] - SOLE_DEPTH - pel[:, None, 2] + self.est[:, None, 2]
    return sole.min(-1).values

  def known_share(self, env) -> torch.Tensor:
    """Share of scan points whose nearest cell is mapped (diagnostics)."""
    robot = env.scene["robot"]
    yaw = _yaw(robot.data.root_link_quat_w) + self.yaw_err
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    ox, oy = self.scan_xy[:, 0], self.scan_xy[:, 1]
    ix, iy = self._cells(torch.stack([self.est[:, :1] + c * ox - s * oy, self.est[:, 1:2] + s * ox + c * oy], -1))
    slot, key = self._slot_key(ix, iy)
    return (self.key.gather(1, slot) == key).float().mean(-1)


class _Sub:
  """Write into a subset of envs (reset prefill)."""

  def __init__(self, vm: VisionMap, ids: torch.Tensor):
    self.vm, self.ids = vm, ids

  def write(self, pts: torch.Tensor, valid: torch.Tensor) -> None:
    vm, ids = self.vm, self.ids
    h, key, est = vm.h, vm.key, vm.est
    vm.h, vm.key, vm.est = h[ids], key[ids], est[ids]
    try:
      vm._write(pts, valid, replace=True)
      h[ids], key[ids] = vm.h, vm.key
    finally:
      vm.h, vm.key, vm.est = h, key, est


def vision_scan(env) -> torch.Tensor:
  """Actor height scan from the onboard map (same layout as stairs_scan)."""
  vm = getattr(env, "_vision", None)
  if vm is None:
    vm = env._vision = VisionMap(env)
  vm.step(env)
  return vm.read(env)
