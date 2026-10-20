"""How far a policy went from the BWC toward the human style, walking forward.

  MUJOCO_GL=egl uv run python scripts/tools/style_compare.py <out.mp4> <run> <ckpt> [<run> <ckpt> ...]

Same style numbers on the BWC reference, each policy (mjlab rollout: forward command,
30 ms delay, measured plant from the environment, 8 identical envs), the training
target (docs/bwc_motion_v3_global_safe.npz) and the plain human (100STYLE Neutral FW,
retarget cache, played x2 slower). Per number: % of the way from the BWC (0) to the
human (100). Video: side views of the BWC, each policy and the human, in step.
"""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import mediapy
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g  # noqa: E402
import retarget_lafan as rt  # noqa: E402

M = rt.model()
JN = [M.joint(j).name for j in range(1, M.njnt)]
DT = 0.005


def ankle_z(q):
  d = mujoco.MjData(M)
  ids = [M.body(f"{s}_ANKLE_P_LINK").id for s in ("L", "R")]
  z = np.zeros((len(q), 2))
  for i, x in enumerate(q):
    d.qpos[:] = x
    mujoco.mj_kinematics(M, d)
    z[i] = d.xpos[ids, 2]
  return z


def metrics(q, dt):
  """q (T, 37) at dt. Forward walking, steady part."""
  j = {n: 7 + k for k, n in enumerate(JN)}
  z = ankle_z(q)
  base = z.min()
  stance = z < base + 0.01
  kn = np.degrees(q[:, j["L_KNEE_P"]])
  w = int(1.0 / dt)
  win = lambda x: np.median([np.ptp(x[i:i + w]) for i in range(0, len(x) - w, w // 2)])  # noqa: E731
  e = Rot.from_quat(q[:, 3:7], scalar_first=True).as_euler("ZYX", degrees=True)
  yaw = np.unwrap(np.radians(e[:, 0]))
  k = max(3, int(1.0 / dt))
  yaw_d = np.degrees(yaw - np.convolve(np.pad(yaw, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(yaw)])
  lift = (z[:, 0] - base)
  td = np.sum(np.diff(stance[:, 0].astype(int)) == 1) + np.sum(np.diff(stance[:, 1].astype(int)) == 1)
  return {
    "genou en appui (deg)": np.median(kn[stance[:, 0]]),
    "genou en vol, max (deg)": np.percentile(kn[~stance[:, 0]], 95) if (~stance[:, 0]).any() else np.nan,
    "bras, balancement (deg)": win(np.degrees(q[:, j["L_SHOULDER_P"]])),
    "buste, rotation (deg)": win(np.degrees(q[:, j["CHEST_Y"]])),
    "bassin, montee-descente (cm)": win(q[:, 2]) * 100,
    "bassin, rotation (deg)": win(yaw_d),
    "bassin, roulis (deg)": win(e[:, 2]),
    "pied, levee (cm)": np.percentile(lift, 95) * 100,
    "pas par seconde": td / (len(q) * dt),
  }


def reference(path, t0=14.0, t1=24.0, clip=0):
  z = np.load(path)
  a = int(z["clip_starts"][clip] + t0 * 200)
  b = int(z["clip_starts"][clip] + t1 * 200)
  return np.concatenate([z["body_pos_w"][a:b, 0], z["body_quat_w"][a:b, 0], z["joint_pos"][a:b]], 1)


def human():
  z = np.load(os.path.expanduser("~/datasets/100style/retarget/Neutral_FW.npz"))
  q, dt = z["qpos"], float(z["dt"])
  # the straightest 10 s, played x2 slower like the training target
  R = Rot.from_quat(q[:, 3:7], scalar_first=True)
  yaw = np.unwrap(R.as_euler("ZYX")[:, 0])
  n = int(5.0 / dt)
  best = min(range(0, len(q) - n, int(1 / dt)), key=lambda a: np.ptp(yaw[a:a + n]))
  return q[best:best + n], dt * 2


def rollout(run_dir, ck, secs=16.0):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  N = 8
  cfg = load_env_cfg(g.TASK, play=True)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = g.MOTION
  lag = int(round(30 / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
  cmd = torch.zeros(N, 3)
  filt = {"y": torch.zeros(N, 3), "step": -1}

  def _cmd(env, command_name, tau=0.0):
    step = int(env.common_step_counter)
    if filt["step"] != step:
      filt["y"] = filt["y"] + (env.step_dt / 0.3) * (cmd - filt["y"])
      filt["step"] = step
    return filt["y"].to(env.device)

  cfg.observations["actor"].terms["velocity_command"].func = _cmd
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
  u = env.unwrapped
  runner = load_runner_cls(g.TASK)(env, asdict(load_rl_cfg(g.TASK)), device="cpu")
  runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run_dir / ck), load_cfg={"actor": True},
              strict=True, map_location="cpu")
  pol = runner.get_inference_policy(device="cpu")
  robot = u.scene["robot"]
  obs, _ = env.reset()
  rec = []
  with torch.no_grad():
    for k in range(int(secs / DT)):
      cmd[:] = torch.tensor([1.0, 0, 0]) if k * DT >= 3.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if k * DT >= 1.0 else torch.zeros(N, env.num_actions))
      if k * DT >= 6.0:
        d = robot.data
        rec.append(np.concatenate([d.root_link_pos_w[0].numpy(), d.root_link_quat_w[0].numpy(),
                                   d.joint_pos[0].numpy()]))
  return np.array(rec)


def main():
  out, args = sys.argv[1], sys.argv[2:]
  rows = [("BWC", reference("docs/bwc_motion_v3.npz"), 0.005)]
  for a in range(0, len(args), 2):
    rows.append((f"{args[a][-8:]} {args[a + 1]}", rollout(args[a], args[a + 1]), DT))
  rows.append(("cible", reference("docs/bwc_motion_v3_global_safe.npz"), 0.005))
  hq, hdt = human()
  rows.append(("humain", hq, hdt))
  ms = [(name, metrics(q, dt)) for name, q, dt in rows]
  keys = list(ms[0][1])
  b, h = ms[0][1], ms[-1][1]
  print("%-30s" % "" + "".join("%18s" % n[:18] for n, _ in ms))
  for k in keys:
    line = "%-30s" % k
    for name, m in ms:
      v = m[k]
      pct = 100 * (v - b[k]) / (h[k] - b[k]) if abs(h[k] - b[k]) > 1e-6 else np.nan
      line += "%10.2f (%4.0f%%)" % (v, pct) if name not in ("BWC", "humain") else "%18.2f" % v
    print(line)
  # video: 8 s of each, side views at 30 fps
  cols = []
  for name, q, dt in rows:
    step = max(1, int(round((1 / 30) / dt)))
    qq = q[::step][:240] if dt < 1 / 30 else q[:240]
    if len(qq) < 240:
      qq = np.concatenate([qq] * (240 // len(qq) + 1))[:240]
    cols.append(rt.panel(M, qq, name, az=100, size=(540, 380)))
  mediapy.write_video(out, np.concatenate(cols, 2), fps=30)
  print("video", out)


if __name__ == "__main__":
  main()
