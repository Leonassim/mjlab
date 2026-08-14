"""Film the training env and report how far the robot actually gets.

Leo reads the wandb videos as a walking robot; every bench says the median
environment does not translate. The two claims have never been put on the same
clip. This resumes the training pipeline exactly as train.py builds it --
VideoRecorder around the training env, same wrapper, same runner -- and prints
Metrics/progress_ratio over the window it filmed.

The curriculum stage 0 is overwritten with the final ranges, so the commands
match the end of the original run without replaying it.

  uv run python scripts/tools/video_with_progress.py <checkpoint.pt> [iterations]
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

TASK = "Mjlab-Velocity-Flat-RHPS1"
NUM_ENVS = 1024
OUT = Path("/tmp/rhps1_video_progress")

FINAL_STAGE = {
  "step": 0,
  "lin_vel_x": (-0.3, 0.3),
  "lin_vel_y": (-0.4, 0.4),
  "ang_vel_z": (-0.45, 0.45),
}

# progress_ratio projects on the command whatever it is, so a mixed command
# distribution can score on lateral or yaw tracking rather than on advancing.
# --forward-only strips the other two axes.
FORWARD_ONLY = {
  "step": 0,
  "lin_vel_x": (0.1, 0.3),
  "lin_vel_y": (0.0, 0.0),
  "ang_vel_z": (0.0, 0.0),
}


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  args = [a for a in sys.argv[2:] if not a.startswith("--")]
  iters = int(args[0]) if args else 12

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=False)
  cfg.scene.num_envs = NUM_ENVS
  stage = FORWARD_ONLY if "--forward-only" in sys.argv else FINAL_STAGE
  if cfg.curriculum and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [stage]
  print(f"[INFO] etage de commande : {stage}")

  agent_cfg = load_rl_cfg(TASK)
  env_raw = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")

  collected: dict[str, list[float]] = {}
  raw_step = env_raw.step

  def instrumented(actions):
    out = raw_step(actions)
    for k, v in env_raw.extras.get("log", {}).items():
      if "progress" in k or k.endswith("command_speed"):
        collected.setdefault(k, []).append(float(v))
    return out

  env_raw.step = instrumented  # type: ignore[method-assign]

  OUT.mkdir(parents=True, exist_ok=True)
  env_v = VideoRecorder(
    env_raw,
    video_folder=OUT / "videos",
    step_trigger=lambda step: step == 0,
    video_length=600,
    disable_logger=True,
  )
  env = RslRlVecEnvWrapper(env_v, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  d = asdict(agent_cfg)
  d["logger"] = "tensorboard"
  runner = runner_cls(env, d, str(OUT / "log"), device)
  runner.load(ckpt, map_location=device)
  runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)

  n = len(collected.get("Metrics/progress_ratio", []))
  print(f"\n{n} pas instrumentes")
  for k in sorted(collected):
    a = np.array(collected[k])
    print(f"  {k:34s} moyenne {a.mean():+.4f}   p50 {np.median(a):+.4f}")
  # Debut contre fin : si le chiffre monte, c'est PPO qui s'adapte pendant la
  # mesure, et le checkpoint lui-meme ne marche pas.
  ra = np.array(collected.get("Metrics/progress_ratio", []))
  if ra.size >= 96:
    w = ra.size // 6
    print("\n  progress_ratio par sixieme de fenetre :")
    print("   " + "  ".join(f"{ra[i * w:(i + 1) * w].mean():+.3f}" for i in range(6)))
  vids = sorted((OUT / "videos").glob("*.mp4"))
  print("\nvideos :", [str(v) for v in vids] or "aucune")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
