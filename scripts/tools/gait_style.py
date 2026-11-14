"""How human does a checkpoint walk? Style numbers next to the human and the BWC.

  uv run python scripts/tools/gait_style.py <run> <ckpt.pt> [<run> <ckpt.pt> ...]

Per command (forward, lateral right, yaw right; 8 envs each, 30 ms delay, filtered
command, plant from the environment as usual), over 10 s of walking: speed,
cadence, pelvis height oscillation (median peak-to-peak per step), stance and
swing knee flexion, arm swing (shoulder pitch range) and ankle lift.
Reference (retarget_lafan.py, 2026-10-19): human slow walk pelvis 5-10 cm,
swing knee 67-83 deg, stance 23-28; BWC 1.9 cm, 56, 30.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/tools")
import gait_dynamics as g  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

CMDS = {"avant": (1, 0, 0), "cote D": (0, -1, 0), "lacet D": (0, 0, -1)}
PER = 8


def run(run_dir, ck):
  N = PER * len(CMDS)
  cfg = load_env_cfg(g.TASK, play=True)
  cfg.scene.num_envs = N
  for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations.pop(k, None)
  cfg.commands["motion"].motion_file = g.MOTION
  lag = int(round(30 / 1000 / cfg.sim.mujoco.timestep))
  for a in cfg.scene.entities["robot"].articulation.actuators:
    a.delay_min_lag = lag
    a.delay_max_lag = lag
  target = torch.tensor([c for c in CMDS.values() for _ in range(PER)], dtype=torch.float32)
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
  jn = list(robot.joint_names)
  bn = list(robot.body_names)
  ia = [bn.index("L_ANKLE_P_LINK"), bn.index("R_ANKLE_P_LINK")]
  rec = {k: [] for k in ("q", "root", "yaw", "ank")}
  obs, _ = env.reset()
  dt = u.step_dt
  with torch.no_grad():
    for k in range(int(16.0 / dt)):
      cmd[:] = target if k * dt >= 3.0 else 0.0
      obs, _, _, _ = env.step(pol(obs) if k * dt >= 1.0 else torch.zeros(N, env.num_actions))
      if k * dt >= 6.0:
        d = robot.data
        rec["q"].append(d.joint_pos.numpy().copy())
        rec["root"].append(d.root_link_pos_w.numpy().copy())
        qw = d.root_link_quat_w.numpy()
        rec["yaw"].append(np.arctan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]),
                                     1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2)))
        rec["ank"].append(d.body_link_pos_w[:, ia].numpy().copy())
  r = {k: np.stack(v, 1) for k, v in rec.items()}  # (N, T, ...)
  T = r["q"].shape[1] * dt
  out = {}
  for i, name in enumerate(CMDS):
    es = range(i * PER, (i + 1) * PER)
    sp, cad, bob, kst, ksw, arm, lift = [], [], [], [], [], [], []
    for e in es:
      root, q, ank = r["root"][e], r["q"][e], r["ank"][e]
      yaw = np.unwrap(r["yaw"][e])
      dp = root[-1, :2] - root[0, :2]
      c, s = np.cos(yaw[0]), np.sin(yaw[0])
      sp.append((dp[0] * c + dp[1] * s, -dp[0] * s + dp[1] * c, np.degrees(yaw[-1] - yaw[0])))
      z = ank[:, :, 2]
      low = z < np.percentile(z, 20) + 0.01  # in stance
      sw = np.diff(low.astype(int), axis=0) == -1  # lift-offs
      cad.append(sw.sum() / T)
      # pelvis bob: peak-to-peak of root z in 0.5 s windows
      w = int(0.5 / dt)
      bob.append(np.median([np.ptp(root[a:a + w, 2]) for a in range(0, len(root) - w, w)]))
      for side, S in ((0, "L"), (1, "R")):
        kn = np.degrees(q[:, jn.index(f"{S}_KNEE_P")])
        kst.append(np.median(kn[low[:, side]]))
        ksw.append(np.percentile(kn[~low[:, side]], 95) if (~low[:, side]).any() else np.nan)
        arm.append(np.degrees(np.ptp(q[:, jn.index(f"{S}_SHOULDER_P")])))
        lift.append(np.percentile(z[:, side] - z.min(), 95))
    sp = np.median(np.array(sp), 0)
    out[name] = (sp, np.median(cad), np.median(bob) * 100, np.median(kst), np.nanmedian(ksw),
                 np.median(arm), np.median(lift) * 100)
  return out


def main():
  args = sys.argv[1:]
  for a in range(0, len(args), 2):
    res = run(args[a], args[a + 1])
    print(f"== {args[a]} {args[a + 1]}")
    for name, (sp, cad, bob, kst, ksw, arm, lift) in res.items():
      print(f"  {name:8s} vx {sp[0] / 10:+.2f} vy {sp[1] / 10:+.2f} m/s lacet {sp[2] / 10:+4.0f} deg/s | "
            f"pas/s {cad:.2f} | bassin {bob:4.1f} cm | genou appui {kst:4.0f} vol {ksw:4.0f} deg | "
            f"bras {arm:4.0f} deg | levee {lift:4.1f} cm", flush=True)


if __name__ == "__main__":
  main()
