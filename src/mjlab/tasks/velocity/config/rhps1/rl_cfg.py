"""RL configuration for RHPS1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
)
from mjlab.tasks.velocity.config.rhps1.rl_ext import RhpsPpoAlgorithmCfg


def rhps1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for RHPS1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # 0.43 (was 2.0, 2026-07-24): leg action scale went 1.5 -> 7.0
        # (rhps1_constants.py's _LEG_SCALE_MULTIPLIER) so ample deliberate
        # excursions are reachable without walking the raw-action mean far
        # from 0 -- but that also means the SAME raw exploration noise now
        # swings the physical joint target ~4.7x further. At init_std=2.0
        # this made even a random policy's exploration alone command
        # violent, uncontrollable position targets from iteration 0 (fell
        # ~10-18/env immediately, never recovering, in a run that reached
        # trackv ~2.5-3.0 cleanly before this scale change). 2.0/(7.0/1.5)
        # keeps the physical-space exploration amplitude the same as
        # before the scale change; only the reachable mean grew.
        "init_std": 0.4286,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      obs_normalization=True,
      hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RhpsPpoAlgorithmCfg(
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
      # Graft kept from the held-noise campaign: the actor must map mirrored
      # observations to mirrored actions. Structurally forbids the
      # one-leg-strides gaits that plagued every white-noise run (foot speed
      # ratios 1.5-1.8); orthogonal to the noise/reward design. See rl_ext.py.
      symmetry_cfg={
        "data_augmentation_func": (
          "mjlab.tasks.velocity.config.rhps1.rl_ext:rhps1_mirror"
        ),
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 1.0,
      },
      class_name="mjlab.tasks.velocity.config.rhps1.rl_ext:TorqueGuidedPPO",
      # Disabled for now (coef=0.0 -> identical to stock PPO, see
      # TorqueGuidedPPO docstring). Four safeguards in a row (cap, warmup,
      # lower ceiling, advantage-mask + Huber) each slowed the same
      # fell_down/trackv-collapse shape without eliminating it -- unclear
      # how much of that is this mechanism versus something else (e.g.
      # stride_frequency_target pushing toward longer, riskier strides).
      # Parking torque feasibility entirely to get a clean read on
      # air_time/stride_frequency_target alone first; pd_demand_excess
      # (see env_cfgs.py) still logs Metrics/pd_demand_ratio for
      # hardware-readiness tracking regardless.
      torque_guidance_coef=0.0,
      torque_guidance_warmup_updates=1000,
    ),
    experiment_name="rhps1_velocity",
    save_interval=150,
    num_steps_per_env=48,
    max_iterations=15_000,
  )
