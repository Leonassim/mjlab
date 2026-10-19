"""Retarget a LAFAN1 BVH segment to RHPS1, kinematically, to look at it before training.

  MUJOCO_GL=egl uv run python scripts/tools/retarget_lafan.py <clip.bvh> <t0> <t1> <out_prefix> [<bwc_clip> <bwc_t0>]

Uniform scale on leg length, then per frame damped least-squares IK on the RHPS1
model (free base + 30 joints, limits clamped), warm-started from the previous frame.
Targets: hip, knee and ankle centres (position), pelvis, chest and feet
(orientation), arms by direction from the robot's own shoulder, head level.
Writes <out>.csv (csv_to_npz format: base pos, quat xyzw, 30 joints in model order),
<out>.npz (qpos at the clip rate) and <out>.mp4: side views of the retargeted
robot and the scaled human, after a docs/bwc_motion_v3.npz clip if one is given.
"""

import sys

import mediapy
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import lafan_bvh as bvh  # noqa: E402

XML = "src/mjlab/asset_zoo/robots/RHPS1/xmls/RHPS1main.xml"
SOLE_DEPTH = 0.097  # ankle centre above the sole, RHPS1 sole box
BONES = [("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"),
         ("Neck", "Head"), ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
         ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"), ("Spine2", "RightShoulder"),
         ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"),
         ("RightForeArm", "RightHand"), ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
         ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToe"), ("Hips", "RightUpLeg"),
         ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToe")]


def model():
  spec = mujoco.MjSpec.from_file(XML)
  w = spec.worldbody
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                   rgb1=[0.78, 0.8, 0.83], rgb2=[0.62, 0.65, 0.7])
  spec.add_material(name="grid", texrepeat=[4, 4], reflectance=0.05).textures[
    mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
  w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", group=2)
  w.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
  spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
  spec.visual.global_.offwidth = 960
  spec.visual.global_.offheight = 720
  return spec.compile()


def _rot_y(a):
  c, s = np.cos(a), np.sin(a)
  R = np.zeros(a.shape + (3, 3))
  R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 1, 1] = 1; R[..., 2, 0] = -s; R[..., 2, 2] = c
  return R


def _level_pitch(R, mask=None):
  """Rotate each frame about its own y so the clip-median pitch of its x axis is 0."""
  pitch = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
  off = np.median(pitch if mask is None else pitch[mask])
  return R @ _rot_y(np.full(len(R), -off))


def human_targets(path, t0, t1):
  names, pos, rot, dt = bvh.fk(path)
  ix = {n: k for k, n in enumerate(names)}
  a, b = int(t0 / dt), int(t1 / dt)
  pos, rot = pos[a:b], rot[a:b]
  P = lambda n: pos[:, ix[n]]  # noqa: E731
  leg = np.median(np.linalg.norm(P("LeftLeg") - P("LeftUpLeg"), axis=-1)
                  + np.linalg.norm(P("LeftFoot") - P("LeftLeg"), axis=-1))
  s = 0.72 / leg
  ankle_h = []
  out = {"dt": dt, "scale": s}
  for side in ("Left", "Right"):
    toe, ft = P(side + "Toe"), P(side + "Foot")
    # stance: foot low and slow
    v = np.linalg.norm(np.gradient(ft, dt, axis=0), axis=-1)
    st = (v < 0.25) & (ft[:, 2] < np.percentile(ft[:, 2], 40))
    ankle_h.append(np.median(ft[st, 2]))
    # lateral axis of the BVH foot frame: the local column most horizontal in stance
    cols = [rot[:, ix[side + "Foot"], :, k] for k in range(3)]
    k = int(np.argmin([np.median(np.abs(c[st, 2])) for c in cols[1:]])) + 1
    lat = cols[k] * np.sign(np.median(np.sum(cols[k] * np.cross([0, 0, 1.0], toe - ft), -1)))
    Rf = bvh.frame(toe - ft, lat)
    out[side + "_foot_R"] = _level_pitch(Rf, st)
    out[side + "_stance"] = st
  dz = SOLE_DEPTH - s * np.mean(ankle_h)
  for n in names:
    out[n] = pos[:, ix[n]] * s + np.array([0, 0, dz])
  out["pelvis_R"] = _level_pitch(bvh.frame(np.cross(P("LeftUpLeg") - P("RightUpLeg"),
                                                    P("Spine2") - P("Hips")),
                                           P("LeftUpLeg") - P("RightUpLeg")))
  out["chest_R"] = _level_pitch(bvh.frame(np.cross(P("LeftArm") - P("RightArm"),
                                                   P("Neck") - P("Spine1")),
                                          P("LeftArm") - P("RightArm")))
  head_x = out["chest_R"][:, :, 0].copy()
  head_x[:, 2] = 0
  out["head_R"] = bvh.frame(head_x, np.cross([0, 0, 1.0], head_x))
  return out


class IK:
  def __init__(self, m):
    self.m, self.d = m, mujoco.MjData(m)
    self.b = {n: m.body(n).id for n in [m.body(i).name for i in range(m.nbody)]}
    self.lo = m.jnt_range[1:, 0].copy()
    self.hi = m.jnt_range[1:, 1].copy()
    self.reg = [m.joint(n).dofadr[0] for n in (
      "L_ELBOW_Y", "R_ELBOW_Y", "L_WRIST_R", "L_WRIST_Y", "R_WRIST_R", "R_WRIST_Y")]

  def _pos(self, body, target, w, J, e):
    jp = np.zeros((3, self.m.nv))
    mujoco.mj_jacBody(self.m, self.d, jp, None, self.b[body])
    J.append(w * jp); e.append(w * (target - self.d.xpos[self.b[body]]))

  def _rot(self, body, Rt, w, J, e):
    jr = np.zeros((3, self.m.nv))
    mujoco.mj_jacBody(self.m, self.d, None, jr, self.b[body])
    Rc = self.d.xmat[self.b[body]].reshape(3, 3)
    J.append(w * jr); e.append(w * Rot.from_matrix(Rt @ Rc.T).as_rotvec())

  def solve(self, h, i, iters=30):
    m, d = self.m, self.d
    for _ in range(iters):
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      J, e = [], []
      self._rot("BODY", h["pelvis_R"][i], 1.0, J, e)
      self._rot("CHEST_P_LINK", h["chest_R"][i], 0.6, J, e)
      self._rot("HEAD_P_LINK", h["head_R"][i], 0.2, J, e)
      for S, s in (("L", "Left"), ("R", "Right")):
        self._pos(f"{S}_CROTCH_P_LINK", h[s + "UpLeg"][i], 3.0, J, e)
        self._pos(f"{S}_KNEE_P_LINK", h[s + "Leg"][i], 2.0, J, e)
        self._pos(f"{S}_ANKLE_P_LINK", h[s + "Foot"][i], 5.0, J, e)
        self._rot(f"{S}_ANKLE_P_LINK", h[s + "_foot_R"][i], 1.0, J, e)
        sh = d.xpos[self.b[f"{S}_SHOULDER_R_LINK"]]
        u = h[s + "ForeArm"][i] - h[s + "Arm"][i]
        f = h[s + "Hand"][i] - h[s + "ForeArm"][i]
        el = sh + 0.261 * u / np.linalg.norm(u)
        self._pos(f"{S}_ELBOW_P_LINK", el, 1.0, J, e)
        self._pos(f"{S}_WRIST_R_LINK", el + 0.23 * f / np.linalg.norm(f), 0.7, J, e)
      for k in self.reg:
        row = np.zeros((1, m.nv)); row[0, k] = 0.3
        J.append(row); e.append(np.array([-0.3 * d.qpos[k + 1]]))
      J, e = np.vstack(J), np.concatenate(e)
      dv = np.linalg.solve(J.T @ J + 1e-3 * np.eye(m.nv), J.T @ e)
      mujoco.mj_integratePos(m, d.qpos, dv, 1.0)
      d.qpos[7:] = np.clip(d.qpos[7:], self.lo, self.hi)
      if np.linalg.norm(dv) < 1e-4:
        break
    return d.qpos.copy(), float(np.linalg.norm(e))


def sole_min_z(m, d):
  z = []
  for side in ("L", "R"):
    g = [k for k in range(m.ngeom) if m.geom_bodyid[k] == m.body(f"{side}_ANKLE_P_LINK").id
         and m.geom_type[k] == mujoco.mjtGeom.mjGEOM_BOX][0]
    hs = m.geom_size[g]
    c = np.array([[sx, sy, -1] for sx in (-1, 1) for sy in (-1, 1)]) * hs
    z.append((d.geom_xpos[g] + c @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
  return z


def panel(m, qs, label, human=None, robot=True, az=110.0, size=(540, 480)):
  """Side view tracking the base (or the human hips), camera heading fixed at the start."""
  from PIL import Image, ImageDraw, ImageFont

  d = mujoco.MjData(m)
  r = mujoco.Renderer(m, *size)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation = 3.0, -14
  opt = mujoco.MjvOption()
  if not robot:
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1
  R0 = Rot.from_quat(qs[0, 3:7], scalar_first=True).as_matrix()
  cam.azimuth = np.degrees(np.arctan2(R0[1, 0], R0[0, 0])) + az
  font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
  frames = []
  for i, q in enumerate(qs):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    c = human["Hips"][i] if not robot else q[:3]
    cam.lookat[:] = [c[0], c[1], 0.7]
    r.update_scene(d, cam, scene_option=opt)
    if human is not None and not robot:
      for a, b in BONES:
        g = r.scene.geoms[r.scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.array([0.2, 0.45, 0.85, 1], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.025, human[a][i], human[b][i])
        r.scene.ngeom += 1
    im = Image.fromarray(r.render())
    ImageDraw.Draw(im).text((14, 10), label, fill=(255, 255, 255), font=font)
    frames.append(np.asarray(im))
  return np.array(frames)


def bwc_qpos(k, t0, n, fps_out):
  """n frames of BWC clip k from t0 at fps_out, looped over its steady part."""
  z = np.load("docs/bwc_motion_v3.npz")
  f = float(z["fps"][0])
  a = int(z["clip_starts"][k] + t0 * f)
  idx = a + (np.arange(n) * f / fps_out).astype(int)
  return np.concatenate([z["body_pos_w"][idx, 0], z["body_quat_w"][idx, 0], z["joint_pos"][idx]], 1)


def retarget(m, path, t0, t1):
  """IK over a segment; returns (qpos (T, 37), human targets, sole heights (T, 2),
  ankle positions (T, 2, 3), IK residuals), sole on z = 0 in stance."""
  h = human_targets(path, t0, t1)
  T = len(h["Hips"])
  ik = IK(m)
  d = ik.d
  d.qpos[:3] = h["Hips"][0]
  d.qpos[3:7] = Rot.from_matrix(h["pelvis_R"][0]).as_quat(scalar_first=True)
  d.qpos[7:] = 0
  d.qpos[[m.joint(n).qposadr[0] for n in ("L_KNEE_P", "R_KNEE_P")]] = 0.3
  qs, err = [], []
  for i in range(T):
    q, e = ik.solve(h, i, 60 if i == 0 else 30)
    qs.append(q); err.append(e)
  qs = np.array(qs)
  # ground: stance sole on z = 0
  zf, ank = [], []
  for q in qs:
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    zf.append(sole_min_z(m, d))
    ank.append([d.xpos[m.body(f"{S}_ANKLE_P_LINK").id].copy() for S in ("L", "R")])
  zf, ank = np.array(zf), np.array(ank)
  zmin = zf.min(1)
  dz = -np.median(zmin)
  zf += dz
  qs[:, 2] += dz
  for n in [k for k in h if isinstance(h[k], np.ndarray) and h[k].shape[-1:] == (3,) and h[k].ndim == 2]:
    h[n] = h[n] + np.array([0, 0, dz])
  return qs, h, zf, ank, np.array(err), dz


def main():
  m = model()
  path, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
  qs, h, zf, ank, err, dz = retarget(m, path, t0, t1)
  T = len(qs)
  ik = IK(m)
  zmin = zf.min(1) - dz
  # report
  jn = [m.joint(j).name for j in range(1, m.njnt)]
  qd = np.gradient(qs[:, 7:], h["dt"], axis=0)
  print(f"scale {h['scale']:.3f}, {T} frames, IK residual median {np.median(err):.3f} max {np.max(err):.3f}, sole z shift {dz:+.3f}")
  near = [(n, np.degrees(qs[:, 7 + j].min()), np.degrees(qs[:, 7 + j].max()))
          for j, n in enumerate(jn) if np.any(qs[:, 7 + j] <= ik.lo[j] + 1e-3) or np.any(qs[:, 7 + j] >= ik.hi[j] - 1e-3)]
  print("butees atteintes:", near if near else "aucune")
  for n in ("L_CROTCH_R", "L_CROTCH_P", "L_KNEE_P", "L_ANKLE_R", "R_ANKLE_R", "L_ANKLE_P", "CHEST_Y", "CHEST_P", "L_SHOULDER_P", "L_ELBOW_P"):
    j = jn.index(n)
    print(f"  {n:13s} [{np.degrees(qs[:, 7 + j].min()):+6.1f}, {np.degrees(qs[:, 7 + j].max()):+6.1f}] deg, |qd| p99 {np.percentile(np.abs(qd[:, j]), 99):.1f} rad/s")
  zs = np.array(zmin) + dz
  va = np.linalg.norm(np.gradient(ank[:, :, :2], h["dt"], axis=0), axis=-1)
  low = zf < 0.01
  print(f"pied au sol (semelle < 1 cm) : glissement median {np.median(va[low]) * 100:.1f} cm/s, p90 {np.percentile(va[low], 90) * 100:.1f} cm/s")
  print(f"semelle la plus basse : p5 {np.percentile(zs, 5) * 100:+.1f} cm, p95 {np.percentile(zs, 95) * 100:+.1f} cm")
  np.savez(out + ".npz", qpos=qs, dt=h["dt"], joint_names=np.array(jn))
  csv = np.concatenate([qs[:, :3], qs[:, [4, 5, 6, 3]], qs[:, 7:]], 1)
  np.savetxt(out + ".csv", csv, delimiter=",", fmt="%.6f")
  fps = round(1 / h["dt"])
  hum = {k: v for k, v in h.items() if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3}
  cols = [panel(m, qs, "RHPS1, marche humaine"),
          panel(m, qs, "humain (LAFAN1, x%.2f)" % h["scale"], human=hum, robot=False)]
  if len(sys.argv) > 6:
    bq = bwc_qpos(int(sys.argv[5]), float(sys.argv[6]), len(qs), fps)
    cols.insert(0, panel(m, bq, "RHPS1, BWC actuel"))
  mediapy.write_video(out + ".mp4", np.concatenate(cols, 2), fps=fps)

if __name__ == "__main__":
  main()
