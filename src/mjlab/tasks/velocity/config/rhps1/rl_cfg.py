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
        # Held (gSDE-lite) exploration noise: each Gaussian draw is kept for
        # hold_steps control steps, moving the exploration variance to low
        # frequency -- ample, trackable joint-space excursions instead of
        # 200 Hz torque-saturating jitter. See rl_ext.py.
        "class_name": (
          "mjlab.tasks.velocity.config.rhps1.rl_ext:HeldNoiseGaussianDistribution"
        ),
        "hold_steps": 16,
        # Back to 1.0 (was 2.0): the doubled std compensated white-noise
        # exploration inefficiency; held noise at 1.0 explores farther.
        "init_std": 1.0,
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
      # Mirror loss: the actor must answer mirrored observations with
      # mirrored actions. Kills the one-leg-strides attractor structurally
      # (it survived the -1/-2/-4 air_time_symmetry penalties, only flipping
      # sides). Data augmentation stays off: height_scan has no mirror rule.
      symmetry_cfg={
        "data_augmentation_func": (
          "mjlab.tasks.velocity.config.rhps1.rl_ext:rhps1_mirror"
        ),
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 1.0,
      },
    ),
    experiment_name="rhps1_velocity",
    save_interval=150,
    num_steps_per_env=48,
    max_iterations=15_000,
  )
