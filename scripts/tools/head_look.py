"""Turn the head toward the walking direction in the lateral clips of a reference.

  uv run python scripts/tools/head_look.py <in.npz> <out.npz> [<deg>=25]

Leo (2026-10-29): sidestepping, a person looks a little where they go. HEAD_Y follows
the pelvis lateral speed in the heading frame, averaged over 4 s (centred, so the head
leads the first step), normalised to a plateau and smoothed; +deg to the left. Only
clips whose net motion is sideways (> 1 m) change. Body poses and velocities are
rebuilt by FK, in the bwc_motion_v3.npz format.
"""

import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import retarget_lafan as rt  # noqa: E402


def box(x, n):
  return np.convolve(np.pad(x, n, mode="edge"), np.ones(2 * n + 1) / (2 * n + 1), "valid")


def main():
  src, out = sys.argv[1], sys.argv[2]
  amp = np.radians(float(sys.argv[3]) if len(sys.argv) > 3 else 25.0)
  z = np.load(src)
  m = rt.model()
  fps = float(np.ravel(z["fps"])[0])
  cs = list(z["clip_starts"]) + [len(z["joint_pos"])]
  q = np.concatenate([z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]], 1)
  hy = m.joint("HEAD_Y").qposadr[0]
  for k, (a, b) in enumerate(zip(cs[:-1], cs[1:])):
    yaw = Rot.from_quat(z["body_quat_w"][a:b, 0], scalar_first=True).as_euler("xyz")[:, 2]
    v = z["body_lin_vel_w"][a:b, 0]
    lat = -np.sin(yaw) * v[:, 0] + np.cos(yaw) * v[:, 1]
    d = z["body_pos_w"][b - 1, 0] - z["body_pos_w"][a, 0]
    if abs(d[1]) < 1.0 or abs(d[1]) < abs(d[0]):
      continue
    vs = box(lat, int(2.0 * fps))
    # Never toward the weight shift that opens and closes a sidestep.
    sgn = np.sign(d[1])
    s = sgn * np.clip(sgn * vs / (0.7 * np.abs(vs).max()), 0, 1)
    s = box(s, int(0.5 * fps))
    q[a:b, hy] = np.clip(q[a:b, hy] + amp * s, *m.jnt_range[m.joint("HEAD_Y").id])
    on = np.abs(s) > 0.05
    print(f"clip {k}: dy {d[1]:+.2f} m, tete {np.degrees(q[a:b, hy]).min():+.0f}..{np.degrees(q[a:b, hy]).max():+.0f} deg, "
          f"tournee {on.mean() * 100:.0f}% du clip")
  d = mujoco.MjData(m)
  pos = np.zeros_like(z["body_pos_w"])
  quat = np.zeros_like(z["body_quat_w"])
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(m, d)
    pos[i], quat[i] = d.xpos[1:], d.xquat[1:]
  dt = 1.0 / fps
  out_d = dict(z)
  out_d["joint_pos"] = q[:, 7:].astype(z["joint_pos"].dtype)
  jv = np.zeros_like(out_d["joint_pos"])
  lv = np.zeros_like(pos)
  av = np.zeros_like(pos)
  for a, b in zip(cs[:-1], cs[1:]):
    jv[a:b] = np.gradient(q[a:b, 7:], dt, axis=0)
    lv[a:b] = np.gradient(pos[a:b], dt, axis=0)
    R = Rot.from_quat(quat[a:b].reshape(-1, 4), scalar_first=True).as_matrix().reshape(b - a, -1, 3, 3)
    dR = np.einsum("tbij,tbkj->tbik", np.gradient(R, dt, axis=0), R)  # dR R^T = [w]x
    av[a:b] = np.stack([dR[..., 2, 1], dR[..., 0, 2], dR[..., 1, 0]], -1)
  out_d.update(joint_vel=jv, body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lv, body_ang_vel_w=av)
  out_d = {k: v.astype(z[k].dtype) for k, v in out_d.items()}
  np.savez_compressed(out, **out_d)
  print("ecrit", out)


if __name__ == "__main__":
  main()
