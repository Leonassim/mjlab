"""LIPM ZMP of a reference vs the stance-foot sole: is the reference dynamically feasible?

uv run python scripts/tools/zmp_check.py <motion.npz> [...]
"""

import sys

import mujoco
import numpy as np

sys.path.insert(0, "scripts/tools")
import retarget_lafan as rt

M = rt.model()
d = mujoco.MjData(M)
ANK = [M.body(f"{s}_ANKLE_P_LINK").id for s in ("L", "R")]
G = [
  [
    k
    for k in range(M.ngeom)
    if M.geom_bodyid[k] == b and M.geom_type[k] == mujoco.mjtGeom.mjGEOM_BOX
  ][0]
  for b in ANK
]
for path in sys.argv[1:]:
  z = np.load(path)
  cs = list(z["clip_starts"]) + [len(z["joint_pos"])]
  for k in (0, 2):
    a, b = cs[k], cs[k + 1]
    q = np.concatenate(
      [z["body_pos_w"][a:b, 0], z["body_quat_w"][a:b, 0], z["joint_pos"][a:b]], 1
    )
    n = len(q)
    c = np.zeros((n, 3))
    az = np.zeros((n, 2))
    gp = np.zeros((n, 2, 3))
    gm = np.zeros((n, 2, 3, 3))
    for i, x in enumerate(q):
      d.qpos[:] = x
      mujoco.mj_kinematics(M, d)
      mujoco.mj_comPos(M, d)
      c[i] = d.subtree_com[1]
      az[i] = d.xpos[ANK, 2]
      gp[i] = d.geom_xpos[G]
      gm[i] = d.geom_xmat[G].reshape(2, 3, 3)
    dt = 1 / 200
    w = 10
    cs_ = np.stack(
      [
        np.convolve(
          np.pad(c[:, j], w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), "valid"
        )
        for j in range(3)
      ],
      1,
    )
    acc = np.gradient(np.gradient(cs_, dt, axis=0), dt, axis=0)
    zmp = cs_[:, :2] - (cs_[:, 2:3] - az.min()) / 9.81 * acc[:, :2]
    st = az < az.min(0) + 0.005
    walk = np.zeros(n, bool)
    walk[int(10 * 200) : n - int(10 * 200)] = True
    ss = walk & (st.sum(1) == 1)
    out, lat, lat_c = [], [], []
    for i in np.where(ss)[0]:
      s = int(np.argmax(st[i]))
      R = gm[i, s]
      hs = M.geom_size[G[s]]
      p = R.T @ (np.r_[zmp[i], gp[i, s, 2]] - gp[i, s])
      pc = R.T @ (cs_[i] - gp[i, s])
      out.append(max(abs(p[0]) - hs[0], abs(p[1]) - hs[1], 0))
      lat.append(p[1])
      lat_c.append(pc[1])
    out = np.array(out) * 100
    print(
      f"{path.split('/')[-1]:28s} clip {k}: appui simple {ss.sum() / walk.sum() * 100:4.0f}% | ZMP hors semelle {np.mean(out > 0) * 100:4.0f}% du temps, p95 {np.percentile(out, 95):4.1f} cm, max {out.max():4.1f} cm"
      f" | CoM lateral / pied p10-p90 {np.percentile(np.abs(lat_c), 10) * 100:4.1f}-{np.percentile(np.abs(lat_c), 90) * 100:4.1f} cm | demi-largeur semelle {M.geom_size[G[0]][1] * 100:.1f} cm"
    )
