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


def fk(path: str):
  """Returns (names, pos (T, J, 3), rot (T, J, 3, 3), dt) for the named joints."""
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
    if p < 0:
      pos[:, j] = lp
      rot[:, j] = local
    else:
      pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", rot[:, p], offsets[j])
      rot[:, j] = rot[:, p] @ local
  keep = [j for j, n in enumerate(names) if n]
  pos = np.einsum("ij,tkj->tki", _M, pos[:, keep]) / 100.0
  rot = _M @ rot[:, keep] @ _M.T
  return [names[j] for j in keep], pos, rot, dt


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
