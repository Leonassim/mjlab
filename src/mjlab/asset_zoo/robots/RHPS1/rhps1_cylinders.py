"""RHPS1 hip and ankle cylinder geometry.

Hip roll/pitch and ankle roll/pitch are each driven by two parallel linear
cylinders (motor, lead screw, RSSR linkage). The low level converts joint targets
to cylinder lengths with the closed form of hrpsys AngleToCylinder.cpp, and the
encoders, motor side, give lengths back. Attachment points from RHPS1.in.conf
(parallel_cylinder_points), link lengths from rhps1-iob rhp_joint_settings.h
(RHPS1 block). Reproduces the conf's neutral lengths (227.494 / 238.428 mm) and
strokes ([194.78, 291.33] / [198.91, 292.79] mm) on the mjlab model.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# name, base_link0, base_link1, pos0, pos2, pos3
CYLINDERS = (
  ("RC-CI", "R_CROTCH_P", "R_CROTCH_Y", (0.0, 0.035, -0.29), (-0.05, 0.035, -0.055), (0.0, 0.035, -0.05)),
  ("RC-CO", "R_CROTCH_P", "R_CROTCH_Y", (0.0, -0.035, -0.29), (-0.05, -0.035, -0.055), (0.0, -0.035, -0.05)),
  ("RC-AI", "R_KNEE_P", "R_ANKLE_P", (0.024, 0.029, -0.075), (-0.04924, 0.029, -0.008682), (0.0, 0.029, -0.317)),
  ("RC-AO", "R_KNEE_P", "R_ANKLE_P", (0.024, -0.029, -0.075), (-0.04924, -0.029, -0.008682), (0.0, -0.029, -0.317)),
  ("LC-CI", "L_CROTCH_P", "L_CROTCH_Y", (0.0, -0.035, -0.29), (-0.05, -0.035, -0.055), (0.0, -0.035, -0.05)),
  ("LC-CO", "L_CROTCH_P", "L_CROTCH_Y", (0.0, 0.035, -0.29), (-0.05, 0.035, -0.055), (0.0, 0.035, -0.05)),
  ("LC-AI", "L_KNEE_P", "L_ANKLE_P", (0.024, -0.029, -0.075), (-0.04924, -0.029, -0.008682), (0.0, -0.029, -0.317)),
  ("LC-AO", "L_KNEE_P", "L_ANKLE_P", (0.024, 0.029, -0.075), (-0.04924, 0.029, -0.008682), (0.0, 0.029, -0.317)),
)
LINK_BC, LINK_BD = 0.07, 0.06

# 42598.4 encoder counts per mm of cylinder, 65536 counts per motor turn.
MOTOR_RAD_PER_M = 2.0 * math.pi * 42598.4e3 / 65536.0

# joint -> (pitch joint, roll joint, cylinders)
PAIRS = {
  f"{s}_{j}_{a}": (f"{s}_{j}_P", f"{s}_{j}_R", tuple(c[0] for c in CYLINDERS if c[0][0] == s and c[0][3] == j[0]))
  for s in "LR" for j in ("CROTCH", "ANKLE") for a in "PR"
}


def cylinder_length(m: mujoco.MjModel, d: mujoco.MjData, cyl, prefix: str = "") -> float:
  _, b0, b1, pos0, pos2, pos3 = cyl
  i0, i1 = m.body(prefix + b0 + "_LINK").id, m.body(prefix + b1 + "_LINK").id
  R0 = d.xmat[i0].reshape(3, 3)
  X, Y, Z = R0.T @ (d.xpos[i1] + d.xmat[i1].reshape(3, 3) @ np.asarray(pos2) - d.xpos[i0]) - np.asarray(pos3)
  BC, BD = LINK_BC, LINK_BD
  if abs(X) > abs(Z):
    alp = (X * X + Z * Z + Y * Y - BC * BC + BD * BD) / (2 * X); bet = -Z / X
  else:
    alp = (X * X + Z * Z + Y * Y - BC * BC + BD * BD) / (2 * Z); bet = -X / Z
  gam = bet * alp; de = alp * alp - BD * BD
  s = math.sqrt(max(gam * gam - (bet * bet + 1) * de, 0.0))
  u0, u1 = (-gam + s) / (bet * bet + 1), (-gam - s) / (bet * bet + 1)
  if abs(X) > abs(Z):
    x0, z0, x1, z1 = alp + bet * u0, u0, alp + bet * u1, u1
  else:
    x0, z0, x1, z1 = u0, alp + bet * u0, u1, alp + bet * u1
  xs, zs = (x0, z0) if x0 < 0 else (x1, z1)
  return float(np.linalg.norm(np.asarray(pos0) - (np.asarray(pos3) + np.array([xs, 0.0, zs]))))


def lever_grid(m: mujoco.MjModel, joint: str, n: int = 33, h: float = 1e-4, prefix: str = ""):
  """Sum over the joint's two cylinders of (dL/dq)^2 on a (pitch, roll) grid
  spanning both joint ranges, normalised to 1 at the neutral pose.

  The drive impedance seen at the joint scales with it: each cylinder's servo
  acts on length, so its joint-side gain goes as (dL/dq)^2.
  """
  jp, jr, names = PAIRS[joint]
  cyls = [c for c in CYLINDERS if c[0] in names]
  d = mujoco.MjData(m)
  ip, ir = m.joint(prefix + jp).id, m.joint(prefix + jr).id
  ap, ar = m.jnt_qposadr[ip], m.jnt_qposadr[ir]
  axis = 0 if joint == jp else 1
  P = np.linspace(*m.jnt_range[ip], n)
  R = np.linspace(*m.jnt_range[ir], n)

  def lengths(p, r):
    d.qpos[:] = m.qpos0; d.qpos[ap] = p; d.qpos[ar] = r
    mujoco.mj_kinematics(m, d)
    return np.array([cylinder_length(m, d, c, prefix) for c in cyls])

  def g(p, r):
    dp = (h, 0.0) if axis == 0 else (0.0, h)
    dl = (lengths(p + dp[0], r + dp[1]) - lengths(p - dp[0], r - dp[1])) / (2 * h)
    return float((dl ** 2).sum())

  G = np.array([[g(p, r) for r in R] for p in P])
  return P, R, G / g(0.0, 0.0)
