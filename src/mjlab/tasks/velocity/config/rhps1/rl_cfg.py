"""RL configuration for RHPS1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def rhps1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for RHPS1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      init_noise_std=1.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      stochastic=True,
      activation="elu",
    ),
    critic=RslRlModelCfg(
      init_noise_std=1.0,
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      stochastic=False,
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="rhps1_velocity",
    save_interval=150,
    num_steps_per_env=48,
    max_iterations=15_000,
  )
