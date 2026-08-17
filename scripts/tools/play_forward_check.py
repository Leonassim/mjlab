"""Does play actually walk, watched over a long window, one real env?

Uses the play=True config (what `uv run play` builds), a forced forward
command injected in _update_command, and a FIXED world camera so travel is
visually unambiguous. Single environment, light on GPU on purpose -- meant to
run alongside a training job.

Prints displacement every few seconds rather than one final number, so a
mid-run fall-and-reset (terminations stay active, as in real play) shows up as
a jump instead of silently corrupting the total.

  uv run python scripts/tools/play_forward_check.py <checkpoint.pt> [speed] [duration_s]
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

TASK = "Mjlab-Velocity-Flat-RHPS1"
POLICY_DT = 0.005
CHECK_EVERY_S = 3.0


def main() -> int:
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  ckpt = sys.argv[1]
  fwd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
  duration_s = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1

  cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  cfg.viewer.lookat = (1.5, 0.0, 0.6)
  cfg.viewer.distance = 7.0
  cfg.viewer.elevation = -12.0
  cfg.viewer.azimuth = 90.0

  env_raw = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env_raw)
  runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=device)
  runner.load(ckpt, map_location=device)
  policy = runner.get_inference_policy(device=device)

  robot = env_raw.scene["robot"]
  cmd = env_raw.command_manager.get_term("twist")
  target = torch.tensor([fwd, 0.0, 0.0], device=device)

  def forced_update() -> None:
    cmd.vel_command_b[:] = target
    cmd.vel_command_w[:] = target
    cmd.vel_command_out[:] = target

  cmd._update_command = forced_update  # type: ignore[method-assign]
  cmd.is_standing_env[:] = False

  env.reset()
  cmd.is_standing_env[:] = False
  obs = env.get_observations()
  if isinstance(obs, tuple):
    obs = obs[0]

  x0 = float(robot.data.root_link_pos_w[0, 0])
  last_check_x = x0
  ep_len0 = int(robot.data.default_root_state[0, 0]) if False else None  # unused

  print(f"commande {fwd} m/s, play=True, camera FIXE, {duration_s:.0f} s\n")
  steps = int(duration_s / POLICY_DT)
  frames = []
  next_check = CHECK_EVERY_S
  t = 0.0
  for i in range(steps):
    with torch.inference_mode():
      action = policy(obs)
    obs = env.step(action)[0]
    cmd.is_standing_env[:] = False
    if i % 4 == 0:
      frames.append(env_raw.render())
    t += POLICY_DT
    if t >= next_check:
      x = float(robot.data.root_link_pos_w[0, 0])
      print(
        f"  t={t:5.1f}s  x total {x - x0:+.3f}m  "
        f"(dernier intervalle {x - last_check_x:+.3f}m sur {CHECK_EVERY_S:.0f}s)"
      )
      last_check_x = x
      next_check += CHECK_EVERY_S

  import mediapy as media

  out = "/tmp/rhps1_play_forward.mp4"
  media.write_video(out, frames, fps=50)
  x_final = float(robot.data.root_link_pos_w[0, 0])
  print(f"\ndeplacement total : {x_final - x0:+.3f} m sur {duration_s:.0f} s")
  print(f"attendu si suivi parfait : {fwd * duration_s:.2f} m")
  print(f"video : {out}")
  env_raw.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
