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
import os

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


# Per drive type: Kt [N.m/A], KP[2] [A/(count/s)], KP[3] [/s], KI[2] [Hz], PL[1] [A].
# RHPS1_drive_data (2026-08-03 export): hip cylinders Drive07/08/12/13, ankle 09/10/14/15.
DRIVES = {
  "CROTCH": (0.246, 1.1e-6, 30.0, 45.0, 2.55),
  "ANKLE": (0.0487, 5.0e-6, 60.0, 2.0, 6.79),
}
COUNTS_PER_M = 42598.4e3


class CylinderDrive:
  """The two cylinder drives of each hip and ankle, as the robot runs them.

  Joint targets go to lengths through the low level's own closed form, each motor
  runs its own P + series PI on its length with its own integral and current
  limit, and the force comes back as J^T F on the motor side of the joint. The
  link hangs on that motor side through the owning actuators' hidden springs; the
  encoder joint angle is the motor side itself (the low level inverts the same
  geometry). Motors massless, backward Euler on the motor angles and integrals.

  The four owning actuators (CROTCH_P/R, ANKLE_P/R) each call compute(); the last
  one of a substep solves the four pairs together and rewrites the earlier ones'
  ctrl, so pitch and roll always see the same instant.
  """

  def __init__(self, entity, mj_model, nworld: int, device: str):
    import torch

    self.entity = entity
    names = list(entity.joint_names)
    prefix = next((mj_model.joint(i).name[: -len(names[0])] for i in range(mj_model.njnt)
                   if mj_model.joint(i).name.endswith(names[0])), "")
    self.pairs = [(s, j) for j in ("CROTCH", "ANKLE") for s in "LR"]
    jid, offR, p0, p2, p3, ankle, C, kp, wi, fmax = ([] for _ in range(10))
    for s, j in self.pairs:
      jid.append([names.index(f"{s}_{j}_P"), names.index(f"{s}_{j}_R")])
      br, bp = mj_model.body(f"{prefix}{s}_{j}_R_LINK"), mj_model.body(f"{prefix}{s}_{j}_P_LINK")
      for b, ax in ((br, (1, 0, 0)), (bp, (0, 1, 0))):
        assert np.allclose(mj_model.body_quat[b.id], (1, 0, 0, 0))
        assert np.allclose(mj_model.jnt_axis[mj_model.body_jntadr[b.id]], ax)
        assert np.allclose(mj_model.jnt_pos[mj_model.body_jntadr[b.id]], 0)
      assert mj_model.body_parentid[bp.id] == br.id and np.allclose(mj_model.body_pos[bp.id], 0)
      offR.append(mj_model.body_pos[br.id])
      cyls = [c for c in CYLINDERS if c[0][0] == s and c[0][3] == j[0]]
      p0.append([c[3] for c in cyls]); p2.append([c[4] for c in cyls]); p3.append([c[5] for c in cyls])
      ankle.append(j == "ANKLE")
      Kt, KP2, KP3, KI, PL = DRIVES[j]
      # RHPS1_DRIVE_GAINS_<CROTCH|ANKLE>="KP3,KI": candidate KP[3], KI[2] [Hz].
      ov = os.environ.get(f"RHPS1_DRIVE_GAINS_{j}")
      if ov:
        KP3, KI = (float(x) for x in ov.split(","))
      C.append([Kt * KP2 * COUNTS_PER_M * MOTOR_RAD_PER_M] * 2)
      kp.append(KP3); wi.append(2.0 * math.pi * KI); fmax.append([Kt * PL * MOTOR_RAD_PER_M] * 2)
    t = lambda x, dt=torch.float64: torch.tensor(np.asarray(x, dtype=float), device=device, dtype=dt)
    self.jid = torch.tensor(jid, device=device, dtype=torch.long)  # (pair, [P, R])
    self.offR = t(offR)
    self.pos0, self.pos2, self.pos3 = t(p0), t(p2), t(p3)  # (pair, cyl, 3)
    self.ankle = torch.tensor(ankle, device=device)
    self.C, self.fmax = t(C), t(fmax)  # (pair, cyl)
    self.kp, self.wi = t(kp)[:, None], t(wi)[:, None]
    shape = (nworld, len(self.pairs), 2)
    z = lambda: torch.zeros(shape, device=device, dtype=torch.float64)
    self.qm, self.integral, self.v = z(), z(), z()
    self.target, self.ks, self.cs, self.tau = z(), z(), z(), z()
    self.scale = torch.ones(shape, device=device, dtype=torch.float64)
    self.fresh = torch.zeros(shape, device=device, dtype=torch.bool)
    self.members: list = []
    self._count = 0
    self.scale_range = (1.0, 1.0)
    self._check(mj_model, prefix)

  def register(self, act) -> None:
    import torch

    slots = []
    for n in act.target_names:
      s, j, a = n.split("_")
      slots.append((self.pairs.index((s, j)), 0 if a == "P" else 1))
    act._cyl_slot = torch.tensor(slots, device=self.jid.device, dtype=torch.long).T
    self.members.append(act)
    self.scale_range = act.cfg.drive_gain_scale_range

  def reset(self, env_ids) -> None:
    import torch

    for b in (self.qm, self.integral, self.v):
      b[env_ids] = 0.0
    lo, hi = self.scale_range
    n = self.scale[env_ids].shape
    self.scale[env_ids] = lo + (hi - lo) * torch.rand(n, device=self.scale.device, dtype=torch.float64)

  def compute(self, act, q_ref, k_s, c_s, fresh, dt: float):
    pi, ji = act._cyl_slot
    self.target[:, pi, ji] = q_ref.double()
    self.ks[:, pi, ji] = k_s.double()
    self.cs[:, pi, ji] = c_s.double()
    self.fresh[:, pi, ji] |= fresh
    self._count += 1
    if self._count == len(self.members):
      self._count = 0
      self._solve(dt)
      for m in self.members:
        if m is not act:
          mp, mj = m._cyl_slot
          self.entity.data.write_ctrl(self.tau[:, mp, mj].float(), m.ctrl_ids)
    return self.tau[:, pi, ji].float()

  def lengths(self, q):
    """Cylinder lengths (env, pair, cyl) at joint angles q (env, pair, [P, R])."""
    import torch

    p, r = q[..., 0, None, None], q[..., 1, None, None]  # (env, pair, 1, 1)
    cp, sp, cr, sr = torch.cos(p), torch.sin(p), torch.cos(r), torch.sin(r)

    def rot(v, sgn):  # Rx(r) Ry(p) v, or its transpose for sgn = -1
      x, y, z = v[..., 0], v[..., 1], v[..., 2]
      if sgn > 0:
        x, z = cp[..., 0] * x + sp[..., 0] * z, -sp[..., 0] * x + cp[..., 0] * z
        y, z = cr[..., 0] * y - sr[..., 0] * z, sr[..., 0] * y + cr[..., 0] * z
      else:
        y, z = cr[..., 0] * y + sr[..., 0] * z, -sr[..., 0] * y + cr[..., 0] * z
        x, z = cp[..., 0] * x - sp[..., 0] * z, sp[..., 0] * x + cp[..., 0] * z
      return torch.stack((x, y, z), -1)

    # Chain origin: the R joint's parent (knee, or hip yaw). P link pose there:
    # R = Rx(r) Ry(p), t = offR (the P joint sits on the R one).
    tr = self.offR[:, None, :]
    a = self.ankle[:, None, None]
    # Ankle: link0 = knee (origin), link1 = P link. Hip: link0 = P link, link1 = hip yaw (origin).
    xyz = torch.where(a, rot(self.pos2.expand(q.shape[0], -1, -1, -1), 1) + tr, rot(self.pos2 - tr, -1)) - self.pos3
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    BC, BD = LINK_BC, LINK_BD
    xz = X.abs() > Z.abs()
    den = torch.where(xz, X, Z)
    alp = (X * X + Z * Z + Y * Y - BC * BC + BD * BD) / (2 * den)
    bet = -torch.where(xz, Z, X) / den
    gam = bet * alp
    de = alp * alp - BD * BD
    s = torch.sqrt(torch.clamp(gam * gam - (bet * bet + 1) * de, min=0.0))
    u0, u1 = (-gam + s) / (bet * bet + 1), (-gam - s) / (bet * bet + 1)
    x0 = torch.where(xz, alp + bet * u0, u0); z0 = torch.where(xz, u0, alp + bet * u0)
    x1 = torch.where(xz, alp + bet * u1, u1); z1 = torch.where(xz, u1, alp + bet * u1)
    xs, zs = torch.where(x0 < 0, x0, x1), torch.where(x0 < 0, z0, z1)
    d = self.pos0 - self.pos3
    return torch.sqrt((d[..., 0] - xs) ** 2 + d[..., 1] ** 2 + (d[..., 2] - zs) ** 2)

  def jacobian(self, q, h: float = 1e-6):
    """dL/dq (env, pair, cyl, [P, R])."""
    import torch

    cols = []
    for k in range(2):
      e = torch.zeros_like(q); e[..., k] = h
      cols.append((self.lengths(q + e) - self.lengths(q - e)) / (2 * h))
    return torch.stack(cols, -1)

  def _solve(self, dt: float) -> None:
    import torch

    data = self.entity.data
    q = data.joint_pos[:, self.jid].double()
    qd = data.joint_vel[:, self.jid].double()
    fresh = self.fresh
    self.qm = torch.where(fresh, self.target, self.qm)
    self.integral = torch.where(fresh.any(-1, keepdim=True), 0.0, self.integral)
    self.fresh = torch.zeros_like(fresh)
    J = self.jacobian(self.qm)
    e0 = self.lengths(self.target) - self.lengths(self.qm)
    C = self.C * self.scale
    fmax = self.fmax * self.scale
    a = self.kp * (1.0 + self.wi * dt)
    b = (1.0 + self.wi * dt) * (1.0 + self.kp * dt)
    D = b * C
    f0 = C * (a * e0 + self.wi * self.integral)
    ks, cs = self.ks, self.cs
    rhs_s = -ks * (self.qm - q) + cs * qd
    diag = ks * dt + cs

    def solve(D, f0):
      # (J^T D J + diag) v = J^T f0 + rhs_s, 2x2 per env and pair
      M = torch.einsum("epci,epc,epcj->epij", J, D, J)
      M = M + torch.diag_embed(diag)
      rhs = torch.einsum("epci,epc->epi", J, f0) + rhs_s
      det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
      v0 = (M[..., 1, 1] * rhs[..., 0] - M[..., 0, 1] * rhs[..., 1]) / det
      v1 = (M[..., 0, 0] * rhs[..., 1] - M[..., 1, 0] * rhs[..., 0]) / det
      v = torch.stack((v0, v1), -1)
      return v, f0 - D * torch.einsum("epci,epi->epc", J, v)

    v, F = solve(D, f0)
    sat = F.abs() > fmax
    if bool(sat.any()):
      # Saturated motors become force sources at their current limit.
      v, F = solve(torch.where(sat, 0.0, D), torch.where(sat, torch.sign(F) * fmax, f0))
    Jv = torch.einsum("epci,epi->epc", J, v)
    ev = self.kp * (e0 - dt * Jv) - Jv
    # Anti-windup: hold the integral while saturated and pushing further.
    self.integral = self.integral + torch.where(sat & (ev * F > 0), 0.0, ev * dt)
    self.qm = self.qm + dt * v
    self.v = v
    self.tau = ks * (self.qm - q) + cs * (v - qd)
    ent = self.entity
    if hasattr(ent, "_rhps1_deflection"):
      ent._rhps1_deflection[:, self.jid] = (self.qm - q).float()
      ent._rhps1_deflection_rate[:, self.jid] = (v - qd).float()

  def _check(self, mj_model, prefix: str) -> None:
    """The torch closed form must match the CPU one on random poses."""
    import torch

    rng = np.random.default_rng(0)
    d = mujoco.MjData(mj_model)
    qs = rng.uniform(-0.4, 0.4, size=(6, len(self.pairs), 2))
    Lt = self.lengths(torch.tensor(qs, device=self.jid.device, dtype=torch.float64)).cpu().numpy()
    for k in range(qs.shape[0]):
      for i, (s, j) in enumerate(self.pairs):
        d.qpos[:] = mj_model.qpos0
        for a, jn in enumerate((f"{s}_{j}_P", f"{s}_{j}_R")):
          d.qpos[mj_model.jnt_qposadr[mj_model.joint(prefix + jn).id]] = qs[k, i, a]
        mujoco.mj_kinematics(mj_model, d)
        cyls = [c for c in CYLINDERS if c[0][0] == s and c[0][3] == j[0]]
        Lc = [cylinder_length(mj_model, d, c, prefix) for c in cyls]
        assert np.allclose(Lt[k, i], Lc, atol=1e-9), (s, j, Lt[k, i], Lc)
