from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import rhps1_flat_env_cfg, rhps1_rough_env_cfg, rhps1_stepping_cfg
from .rl_cfg import rhps1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-RHPS1",
  env_cfg=rhps1_rough_env_cfg(),
  play_env_cfg=rhps1_rough_env_cfg(play=True),
  rl_cfg=rhps1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-RHPS1",
  env_cfg=rhps1_flat_env_cfg(),
  play_env_cfg=rhps1_flat_env_cfg(play=True),
  rl_cfg=rhps1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Stepping-RHPS1",
  env_cfg=rhps1_stepping_cfg(),
  play_env_cfg=rhps1_stepping_cfg(play=True),
  rl_cfg=rhps1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
