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
        # Policy 0's value. What matters is the product init_std * action_scale,
        # so this must be rescaled whenever the leg action scale changes.
        "init_std": 1.0,
        # The cap must stay ABOVE init_std: torch.clamp has exactly zero
        # gradient outside the range in both directions, so a cap at or below
        # init_std freezes std_param instead of bounding it. That bug once held
        # it constant for 289 iterations. Watch Mean action std: it should move.
        "std_range": (1e-6, 1.3),
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
      # Policy 0's value. The entropy bonus gradient on log sigma is exactly
      # this coefficient regardless of action scale, so a scale change does not
      # require restating it -- unlike init_std and std_range.
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
      # 0.3 (was 0.0, 2026-07-28). Re-enabled deliberately, and this is the
      # highest-risk change of the batch -- it was parked because four
      # safeguards in a row each slowed but never eliminated a
      # fell_down/trackv collapse. Two things changed since:
      #   - the named confound (stride_frequency_target pushing toward
      #     longer, riskier strides) has since been removed entirely, so the
      #     "unclear how much is this mechanism" caveat is now testable;
      #   - torque feasibility is currently enforced by *nothing*
      #     (pd_demand_excess was inert at -1e-4), and
      #     Metrics/pd_demand_ratio_mean is 5.6 with a max of 350 -- the
      #     exact failure mode that blew up mc_mujoco on hardware-side
      #     clamping.
      # 0.3 rather than the previous ceiling: enough to bend the demand
      # curve, small enough to abandon cheaply. If fell_down climbs steadily
      # after warmup ends, this is the first thing to switch back off.
      # Back to 0.0 (2026-07-28, same day as the 0.3 attempt above). That
      # attempt reproduced the documented failure exactly: fell_down 0.21 ->
      # 0.74 and trackv 2.63 -> 1.02 starting ~iteration 1200, i.e. right in
      # the "collapsed anyway once the ramp approached full strength
      # (~iteration 900-1200)" window of attempt 2. Note this is now the
      # *fifth* failure of this mechanism, and attempt 3 had already
      # established that lowering the coefficient only slows it -- so
      # retrying at 0.1 would have been re-running a known-negative
      # experiment. The confound I hoped had been removed
      # (stride_frequency_target) evidently was not the cause.
      # Retire le 2026-08-07 : coefficient a 0 apres cinq echecs, donc inerte,
      # mais il maintenait en vie le groupe d'observation torque_guidance.
    ),
    experiment_name="rhps1_velocity",
    save_interval=150,
    num_steps_per_env=48,
    max_iterations=15_000,
  )
