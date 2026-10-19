"""LAFAN1 BVH reader: global joint positions and rotations, Z-up, meters.

  uv run python scripts/tools/lafan_bvh.py <clip.bvh>   (prints a walking timeline)
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# BVH is Y-up: (x, y, z) -> (x, -z, y).
_M = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def read(path: str):
  names, parents, offsets, chans = [], [], [], []
  stack, motion = [], None
  with open(path) as f:
    lines = f.read().split("\n")
  i = 0
  while i < len(lines):
    tok = lines[i].split()
    i += 1
    if not tok:
      continue
    if tok[0] in ("ROOT", "JOINT"):
      names.append(tok[1])
      parents.append(stack[-1] if stack else -1)
      chans.append([])
    elif tok[0] == "End":
      names.append(None)
      parents.append(stack[-1])
      chans.append([])
    elif tok[0] == "{":
      stack.append(len(names) - 1)
    elif tok[0] == "}":
      stack.pop()
    elif tok[0] == "OFFSET":
      offsets.append([float(x) for x in tok[1:4]])
    elif tok[0] == "CHANNELS":
      chans[-1] = tok[2:]
    elif tok[0] == "MOTION":
      n = int(lines[i].split()[1])
      dt = float(lines[i + 1].split()[2])
      motion = np.array([[float(x) for x in ln.split()] for ln in lines[i + 2 : i + 2 + n]])
      break
  return names, np.array(parents), np.array(offsets), chans, motion, dt


def _euler_sxyz(a):
  """ASF/AMC convention: static XYZ, R = Rz Ry Rx; a (..., 3) in degrees."""
  return Rot.from_euler("xyz", a, degrees=True).as_matrix()


def fk_amc(path: str):
  """CMU ASF/AMC (the .asf next to the .amc, named after the subject), 120 Hz."""
  import glob
  import os

  asf = glob.glob(os.path.join(os.path.dirname(path), "*.asf"))[0]
  bones, children, unit = {}, {}, 1.0
  lines = open(asf).read().split("\n")
  i, section, cur = 0, None, None
  for ln in lines:
    t = ln.split()
    if not t or t[0].startswith("#"):
      continue
    if t[0].startswith(":"):
      section = t[0]
      continue
    if section == ":units" and t[0] == "length":
      unit = float(t[1])
    elif section == ":bonedata":
      if t[0] == "begin":
        cur = {"dof": []}
      elif t[0] == "end":
        bones[cur["name"]] = cur
      elif t[0] == "name":
        cur["name"] = t[1]
      elif t[0] == "direction":
        cur["dir"] = np.array([float(x) for x in t[1:4]])
      elif t[0] == "length":
        cur["len"] = float(t[1])
      elif t[0] == "axis":
        cur["axis"] = np.array([float(x) for x in t[1:4]])
      elif t[0] == "dof":
        cur["dof"] = t[1:]
    elif section == ":hierarchy" and t[0] not in ("begin", "end"):
      children[t[0]] = t[1:]
  frames, cur = [], None
  for ln in open(path).read().split("\n"):
    t = ln.split()
    if not t or t[0].startswith(("#", ":")):
      continue
    if len(t) == 1 and t[0].isdigit():
      cur = {}
      frames.append(cur)
    elif cur is not None:
      cur[t[0]] = np.array([float(x) for x in t[1:]])
  T = len(frames)
  m = 0.0254 / unit  # ASF length unit is 1/unit inch
  pos, rot = {}, {}
  root = np.array([f["root"] for f in frames])
  pos["root"] = root[:, :3] * m
  rot["root"] = _euler_sxyz(root[:, 3:6])
  stack = ["root"]
  while stack:
    b = stack.pop()
    for c in children.get(b, []):
      bd = bones[c]
      C = _euler_sxyz(bd["axis"])
      a = np.zeros((T, 3))
      if bd["dof"]:
        v = np.array([f.get(c, np.zeros(len(bd["dof"]))) for f in frames])
        for k, dname in enumerate(bd["dof"]):
          a[:, "xyz".index(dname[1])] = v[:, k]
      R = rot[b] @ C @ _euler_sxyz(a) @ C.T
      rot[c] = R
      pos[c] = pos[b] + bd["len"] * m * np.einsum("tij,j->ti", R, bd["dir"])
      stack.append(c)
  # joint = end of the named bone
  ends = {"Hips": "root", "LeftUpLeg": "lhipjoint", "LeftLeg": "lfemur", "LeftFoot": "ltibia",
          "LeftToe": "lfoot", "RightUpLeg": "rhipjoint", "RightLeg": "rfemur",
          "RightFoot": "rtibia", "RightToe": "rfoot", "Spine": "lowerback", "Spine1": "upperback",
          "Spine2": "thorax", "Neck": "lowerneck", "Head": "upperneck",
          "LeftShoulder": "thorax", "LeftArm": "lclavicle", "LeftForeArm": "lhumerus",
          "LeftHand": "lradius", "RightShoulder": "thorax", "RightArm": "rclavicle",
          "RightForeArm": "rhumerus", "RightHand": "rradius"}
  # rotation of the segment below the joint (the foot for LeftFoot, as in BVH)
  below = {"LeftFoot": "lfoot", "RightFoot": "rfoot", "LeftUpLeg": "lfemur", "RightUpLeg": "rfemur",
           "LeftLeg": "ltibia", "RightLeg": "rtibia"}
  names = list(ends)
  P = np.stack([pos[ends[n]] for n in names], 1)
  Rm = np.stack([rot[below.get(n, ends[n])] for n in names], 1)
  P = np.einsum("ij,tkj->tki", _M, P)
  Rm = _M @ Rm @ _M.T
  return names, P, Rm, 1.0 / 120.0


def fk(path: str):
  """Returns (names, pos (T, J, 3), rot (T, J, 3, 3), dt) for the named joints."""
  if path.endswith(".amc"):
    return fk_amc(path)
  names, parents, offsets, chans, mot, dt = read(path)
  T, J = len(mot), len(names)
  pos = np.zeros((T, J, 3))
  rot = np.zeros((T, J, 3, 3))
  col = 0
  for j in range(J):
    c = chans[j]
    if c:
      v = mot[:, col : col + len(c)]
      col += len(c)
      rc = [k for k, n in enumerate(c) if n.endswith("rotation")]
      order = "".join(c[k][0] for k in rc)
      local = Rot.from_euler(order.upper(), v[:, rc], degrees=True).as_matrix()
      lp = v[:, [k for k, n in enumerate(c) if n.endswith("position")]]
    else:
      local = np.broadcast_to(np.eye(3), (T, 3, 3))
      lp = None
    p = parents[j]
    # position channels, when present, replace the offset (Bandai Namco has them on
    # every joint, LAFAN1 on the root only)
    t = lp if lp is not None and lp.shape[1] == 3 else np.broadcast_to(offsets[j], (T, 3))
    if p < 0:
      pos[:, j] = t
      rot[:, j] = local
    else:
      pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", rot[:, p], t)
      rot[:, j] = rot[:, p] @ local
  keep = [j for j, n in enumerate(names) if n]
  pos = np.einsum("ij,tkj->tki", _M, pos[:, keep]) / 100.0
  rot = _M @ rot[:, keep] @ _M.T
  out = [names[j] for j in keep]
  if "UpperLeg_L" in out:  # Bandai Namco -> LAFAN1 names
    k = [i for i, n in enumerate(out) if n != "joint_Root"]  # Hips first, as in LAFAN1
    out = [_BANDAI.get(out[i], out[i]) for i in k]
    pos, rot = pos[:, k], rot[:, k]
    i = out.index("Spine")  # no Spine1: use Spine
    pos = np.concatenate([pos, pos[:, [i]]], 1)
    rot = np.concatenate([rot, rot[:, [i]]], 1)
    out = out + ["Spine1"]
  elif "LeftCollar" in out:  # 100STYLE -> LAFAN1 names
    out = [_100STYLE.get(n, n) for n in out]
  return out, pos, rot, dt


_100STYLE = {"Chest": "Spine", "Chest2": "Spine1", "Chest4": "Spine2",
             "LeftCollar": "LeftShoulder", "LeftShoulder": "LeftArm", "LeftElbow": "LeftForeArm",
             "LeftWrist": "LeftHand", "LeftHip": "LeftUpLeg", "LeftKnee": "LeftLeg",
             "LeftAnkle": "LeftFoot", "RightCollar": "RightShoulder", "RightShoulder": "RightArm",
             "RightElbow": "RightForeArm", "RightWrist": "RightHand", "RightHip": "RightUpLeg",
             "RightKnee": "RightLeg", "RightAnkle": "RightFoot"}

_BANDAI = {"Chest": "Spine2", "UpperLeg_L": "LeftUpLeg", "LowerLeg_L": "LeftLeg",
           "Foot_L": "LeftFoot", "Toes_L": "LeftToe", "UpperLeg_R": "RightUpLeg",
           "LowerLeg_R": "RightLeg", "Foot_R": "RightFoot", "Toes_R": "RightToe",
           "Shoulder_L": "LeftShoulder", "UpperArm_L": "LeftArm", "LowerArm_L": "LeftForeArm",
           "Hand_L": "LeftHand", "Shoulder_R": "RightShoulder", "UpperArm_R": "RightArm",
           "LowerArm_R": "RightForeArm", "Hand_R": "RightHand"}


def frame(x, y_hint):
  """Right-handed frame with x exact and y from the hint, columns (x, y, z)."""
  x = x / np.linalg.norm(x, axis=-1, keepdims=True)
  y = y_hint - np.sum(y_hint * x, -1, keepdims=True) * x
  y = y / np.linalg.norm(y, axis=-1, keepdims=True)
  return np.stack([x, y, np.cross(x, y)], -1)


def pelvis_frame(names, pos):
  ix = {n: k for k, n in enumerate(names)}
  left = pos[:, ix["LeftUpLeg"]] - pos[:, ix["RightUpLeg"]]
  up = pos[:, ix["Spine"]] - pos[:, ix["Hips"]]
  # x = left x up (forward), then y = left.
  fwd = np.cross(left, up)
  return frame(fwd, left)


def timeline(path: str, win: float = 2.0):
  names, pos, rot, dt = fk(path)
  R = pelvis_frame(names, pos)
  p = pos[:, 0, :2]
  v = np.gradient(p, dt, axis=0)
  yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
  wz = np.gradient(yaw, dt)
  n = int(win / dt)
  print(f"{path.split('/')[-1]}: {len(p) * dt:.0f} s, hips height median {np.median(pos[:, 0, 2]):.2f} m")
  for a in range(0, len(p) - n, n):
    s = slice(a, a + n)
    c, sn = np.cos(yaw[s]), np.sin(yaw[s])
    vf = np.mean(v[s, 0] * c + v[s, 1] * sn)
    vl = np.mean(-v[s, 0] * sn + v[s, 1] * c)
    print(f"  {a * dt:6.1f}s  avant {vf:+.2f}  cote {vl:+.2f} m/s  lacet {np.degrees(np.mean(wz[s])):+5.0f} deg/s")


if __name__ == "__main__":
  timeline(sys.argv[1])
