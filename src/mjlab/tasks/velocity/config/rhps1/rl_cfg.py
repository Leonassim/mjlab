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
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # 2.0 (was 1.0): together with the x4 leg action scale, widens early
        # joint-space exploration so long strides are reachable by sampling.
        "init_std": 2.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
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
