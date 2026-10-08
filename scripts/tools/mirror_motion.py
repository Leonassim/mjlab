"""Append the left-right mirror of a motion clip, as a second clip in one file.

  uv run python scripts/tools/mirror_motion.py docs/bwc_motion_v2.npz docs/bwc_motion_v2_lr.npz

The BWC log walks sideways and turns to the left only; the mirror gives the
right. The output holds both clips and "clip_starts", so the tracking command
resets at the seam instead of teleporting the reference. Checked by forward
kinematics: the mirrored joints must rebuild the mirrored body poses.
"""
import sys, numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.rhps1.rl_ext import _joint_perm_sign
src, dst = sys.argv[1], sys.argv[2]
cfg = load_env_cfg("Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start", play=True)
cfg.scene.num_envs = 1
cfg.commands["motion"].motion_file = src
env = ManagerBasedRlEnv(cfg, device="cpu")
robot = env.scene["robot"]
bn, jn = list(robot.body_names), list(robot.joint_names)
print("bodies", bn)
def swap(n):
  for a, b in (("L_", "R_"), ("left", "right"), ("Left", "Right")):
    if n.startswith(a): return b + n[len(a):]
    if n.startswith(b): return a + n[len(b):]
  return n
bp = [bn.index(swap(n)) for n in bn]
jp, js = _joint_perm_sign(jn)
jp, js = np.asarray(jp), np.asarray(js, dtype=np.float32)
m = dict(np.load(src))
out = dict(m)
out["joint_pos"] = m["joint_pos"][:, jp] * js
out["joint_vel"] = m["joint_vel"][:, jp] * js
out["body_pos_w"] = m["body_pos_w"][:, bp] * np.array([1, -1, 1], np.float32)
out["body_lin_vel_w"] = m["body_lin_vel_w"][:, bp] * np.array([1, -1, 1], np.float32)
out["body_ang_vel_w"] = m["body_ang_vel_w"][:, bp] * np.array([-1, 1, -1], np.float32)
out["body_quat_w"] = m["body_quat_w"][:, bp] * np.array([1, -1, 1, -1], np.float32)
# FK check on 40 frames: write mirrored root + joints, compare body positions.
err, err2, bad = [], [], []
for f in np.linspace(0, len(m["joint_pos"]) - 1, 40).astype(int):
  root = torch.zeros(1, 13)
  root[0, :3] = torch.as_tensor(out["body_pos_w"][f, 0]) + env.scene.env_origins[0]
  root[0, 3:7] = torch.as_tensor(out["body_quat_w"][f, 0])
  robot.write_root_state_to_sim(root)
  robot.write_joint_state_to_sim(torch.as_tensor(out["joint_pos"][f])[None], torch.zeros(1, len(jn)))
  env.sim.forward()
  p = robot.data.body_link_pos_w[0].numpy() - env.scene.env_origins[0].numpy()
  err.append(np.abs(p - out["body_pos_w"][f]).max())
  q = robot.data.body_link_quat_w[0].numpy(); qe = np.minimum(np.abs(q - out["body_quat_w"][f]).max(), np.abs(q + out["body_quat_w"][f]).max()); err2.append(qe)
  bad.append(np.abs(p - m["body_pos_w"][f]).max())
print(f"quat error max {max(err2):.4f} ; negative control (FK vs ORIGINAL clip) {max(bad)*1000:.0f} mm")
print(f"FK mirrored clip: max body position error {max(err)*1000:.2f} mm over 40 frames")
cat = {k: (np.concatenate([m[k], out[k]]) if m[k].ndim > 1 else m[k]) for k in m}
cat["clip_starts"] = np.array([0, len(m["joint_pos"])])
np.savez(dst, **cat)
print("wrote", dst)
