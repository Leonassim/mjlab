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
CAM_POS = (0.071, 0.0017, 0.1558)  # azure_depth_frame in HEAD_P_LINK (URDF), MuJoCo camera axes (-z forward)
CAM_RPY = (1.46608, 1.45688e-06, -1.5708)
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
    self.lat_n = max(0, round(LAT / (env.step_dt * PERIOD)))
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
    pre = ids[torch.rand(k, device=dev) < PREFILL]
    if pre.numel():
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
      self.fifo.append(self._capture(env, pel))
      while len(self.fifo) > self.lat_n:
        p, v = self.fifo.pop(0)
        self._write(p, v, replace=False)

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
