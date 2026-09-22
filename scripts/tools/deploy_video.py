"""Record the DEPLOYMENT condition: standing start, held command, no noise.

Training videos show whatever env 0 was doing -- 50 % of the time an episode
started mid-gait ON the reference, which flatters the policy. This records what
mc_mujoco actually runs.

  uv run python scripts/tools/deploy_video.py <run> <model.pt> [secs] [vx vy wz]
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

TASK = "Mjlab-Tracking-Flat-RHPS1-Copy-V3-Start"
run, ck = sys.argv[1], sys.argv[2]
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
cmd = torch.tensor([float(x) for x in sys.argv[4:7]] if len(sys.argv) > 6 else [1.0, 0.0, 0.0])

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1
for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
  cfg.terminations.pop(k, None)
cfg.commands["motion"].motion_file = "docs/bwc_motion_v2.npz"
cfg.observations["actor"].terms["velocity_command"].func = (
  lambda env, command_name, _c=cmd: _c.to(env.device).repeat(env.num_envs, 1)
)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
out = Path("logs/rsl_rl/rhps1_tracking") / run / "videos" / "deploiement"
env = VideoRecorder(env, video_folder=out, step_trigger=lambda s: s == 0,
                    video_length=int(secs / cfg.sim.render_interval / 0.005) if False else int(secs / 0.005),
                    name_prefix=f"deploiement-{Path(ck).stem}", disable_logger=True)
venv = RslRlVecEnvWrapper(env)
runner = load_runner_cls(TASK)(venv, asdict(load_rl_cfg(TASK)), device="cpu")
runner.load(str(Path("logs/rsl_rl/rhps1_tracking") / run / ck), load_cfg={"actor": True},
            strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
robot = env.scene["robot"]
obs, _ = venv.reset()
q0 = robot.data.default_joint_pos.clone()
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
root = robot.data.default_root_state.clone()
root[:, :3] += env.scene.env_origins
robot.write_root_state_to_sim(root)
with torch.no_grad():
  for _ in range(600):  # 3 s standing, as mc_mujoco does before arming
    obs, _, _, _ = venv.step(torch.zeros(1, venv.num_actions))
  for _ in range(int(secs / 0.005)):
    obs, _, _, _ = venv.step(policy(obs))
env.close()
print("video:", out)
