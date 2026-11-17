"""vismoke.py <run> <ckpt>: onboard map vs exact scan under a deterministic policy (RHPS1_VIS=1 forced).

DRIVE=exact (default) runs the policy on the exact scan and only measures the map; DRIVE=vision feeds it the map.
Prints map error by scan region, known share, ray count, then the apsmoke-style SUMMARY.
"""
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ["RHPS1_VIS"] = "1"
os.environ.setdefault("RHPS1_STAIRS", "1")
import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import stairs as ST  # noqa: E402
from mjlab.tasks.tracking.config.rhps1 import vision_map as VM  # noqa: E402

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
cfg = load_env_cfg(TASK)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v3.npz"
cfg.scene.num_envs = int(os.environ.get("SMOKE_N", "16"))
torch.manual_seed(int(os.environ.get("SEED", "0")))
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"), clip_actions=12.0)
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / sys.argv[1] / sys.argv[2]), load_cfg={"actor": True},
            strict=True, map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
u = env.unwrapped
st, robot = u._stairs, u.scene["robot"]
drive_exact = os.environ.get("DRIVE", "exact") == "exact"
terms = list(u.observation_manager.active_terms["actor"])
dims = u.observation_manager.group_obs_term_dim["actor"]
i0 = sum(math.prod(d) for d in dims[: terms.index("height_scan")])
ns = math.prod(dims[terms.index("height_scan")])
print("rays", u.scene["stairs_cam"].num_rays, "| scan", ns, "at", i0, "| head", round(math.degrees(VM.HEAD_P), 1), "deg",
      "| latency steps", round(VM.LAT / u.step_dt), "| drive", "exact" if drive_exact else "vision")

nx = len(torch.arange(-ST.SCAN_SIZE[0] / 2, ST.SCAN_SIZE[0] / 2 + ST.SCAN_RES * 0.5, ST.SCAN_RES))
xs = u.scene["stairs_scan"]._local_offsets[:, 0]
regions = {"behind": xs < 0.0, "0-0.3": (xs >= 0) & (xs < 0.3), "0.3-0.6": (xs >= 0.3) & (xs < 0.6), "0.6-0.9": xs >= 0.6}


def exact_scan():
  d = u.scene["stairs_scan"].data
  return (d.pos_w[:, None, 2] - d.hit_pos_w[..., 2] - ST.SCAN_OFFSET).clamp(-1, 1)


obs = env.get_observations()
T = int(float(os.environ.get("SMOKE_T", "12")) / u.step_dt)
err_sum = {k: 0.0 for k in regions}
err_max = {k: 0.0 for k in regions}
big = {k: 0 for k in regions}
cnt, known = 0, 0.0
ARR = torch.zeros(u.num_envs)
FALL = torch.zeros(u.num_envs)
t0 = time.time()
for i in range(T):
  o = obs["actor"] if "actor" in obs.keys() else obs
  vis = o[:, i0:i0 + ns].clone()
  ex = exact_scan()
  if i > 2:
    e = (vis - ex).abs()[st.pyr] if os.environ.get("PYR_ONLY", "1") == "1" else (vis - ex).abs()
    for k, m in regions.items():
      err_sum[k] += float(e[:, m].mean())
      err_max[k] = max(err_max[k], float(e[:, m].max()))
      big[k] += int((e[:, m] > 0.03).sum())
    known += float(u._vision.known_share(u).mean())
    cnt += 1
  if i % 100 == 10:
    import warp as wp
    cd = u.scene["stairs_cam"].data
    gid = wp.to_torch(u.scene["stairs_cam"]._ray_geomid).long()
    rob = (gid >= 0) & u._vision.robot_geom[gid.clamp(min=0)]
    ok = (cd.distances >= VM.RANGE[0]) & ~rob & (gid >= 0)
    q = robot.data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    rel = cd.hit_pos_w[..., :2] - robot.data.root_link_pos_w[:, None, :2]
    fx = torch.cos(yaw)[:, None] * rel[..., 0] + torch.sin(yaw)[:, None] * rel[..., 1]
    fxv = fx[ok]
    print(f"camera: valid {float(ok.float().mean()):.2f} robot-hidden {float(rob.float().mean()):.2f} miss {float((gid < 0).float().mean()):.2f}"
          f" | ground seen x {float(fxv.min()):.2f}..{float(fxv.max()):.2f} m ahead of the pelvis")
    rx = fx[rob]
    if rx.numel():
      print(f"  hidden rays aimed between x {float(rx.min()):.2f} and {float(rx.max()):.2f} (hit points on the body)")
  if drive_exact:
    o = o.clone()
    o[:, i0:i0 + ns] = ex
    obs["actor"] = o
  with torch.no_grad():
    obs, *_ = env.step(pol(obs))
  ARR += st.arrived_now.float().cpu()
  if "stairs_fall" in u.termination_manager.active_terms:
    FALL += u.termination_manager.get_term("stairs_fall").float().cpu()
dt_wall = time.time() - t0
pts = {k: int(m.sum()) for k, m in regions.items()}
print("map |err| mean / max (cm), share > 3 cm:",
      " | ".join(f"{k} {100 * err_sum[k] / cnt:.1f}/{100 * err_max[k]:.0f} {big[k] / (cnt * max(1, e.shape[0]) * pts[k]):.1%}"
                 for k in regions))
print(f"known share {known / cnt:.2f} | {dt_wall / T * 1000:.0f} ms/step on CPU")
print(f"SUMMARY {sys.argv[2]} drive {'exact' if drive_exact else 'vision'}: arrivals {int(ARR.sum())} falls {int(FALL.sum())}"
      f" over {u.num_envs} envs x {T * u.step_dt:.0f} s")
