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
        # Re-added (2026-07-27): the 2026-07-16 fix for this exact failure
        # mode (commit 4b9018ee, "entropy bonus inflated std to 1.9 once
        # episodes got too short to carry any other learning signal") fell
        # out of this file at some point and was never rescaled when
        # init_std was recalibrated for the leg-scale change above. Mean
        # action std climbed monotonically 0.43 -> 3.73 over the first 2876
        # iterations of the 2026-07-27 rescaled-penalty run, never
        # plateauing.
        #
        # 1.3 (was 0.27, 2026-07-27, same day): 0.27 was computed by
        # rescaling the OLD 1.25 cap (from the 1.0-init/1.25-cap pair in
        # 4b9018ee) by the current (1.5/7.0) leg-scale factor -- but that
        # 1.25 was never paired with init_std=1.0 at leg_scale=1.5 in the
        # first place; it's from an earlier, unrelated snapshot of this
        # file. Mixing the two produced a cap (0.27) *below* init_std
        # (0.4286): rsl_rl's GaussianDistribution clamps std_param with
        # plain torch.clamp, whose gradient is exactly zero outside the
        # clamped range in *both* directions -- so std_param was frozen at
        # its raw initial value from iteration 0, never receiving any
        # gradient (policy or entropy) at all. Mean action std read exactly
        # 0.27 (and Mean entropy loss exactly constant) for 289 iterations
        # straight on the 2026-07-27 entropy_coef=0.005 rerun -- not slow
        # convergence, a dead parameter.
        #
        # Re-derived in physical units instead, scale-invariant across any
        # future leg_scale change: with std_type="scalar" one std is shared
        # across every joint, so the largest action_scale is the binding
        # constraint -- CROTCH_P at 0.049 rad/raw-unit (verified via
        # rhps1_constants.RHPS1_ACTION_SCALE). The design target this
        # init_std encodes is 0.4286 * 0.049 = 0.021 rad (~1.2 deg) 1-sigma,
        # deliberately unchanged across the leg-scale rescale (2.0 * 0.0105
        # gave the same 0.021 rad before it). The old cap (0.27) gave 0.27 *
        # 0.049 = 0.0132 rad (~0.76 deg) -- *tighter* than the design target
        # itself, on top of being dead. The blown-up run's 3.73 gave 3.73 *
        # 0.049 = 0.183 rad (~10.5 deg), ~8.7x the design target -- the
        # actual runaway that produced the visible vibration/waddle. 1.3
        # gives 1.3 * 0.049 = 0.064 rad (~3.6 deg), ~3x the design target:
        # real headroom to explore above init, nowhere near the 8.7x that
        # was visibly destabilizing, and safely above init_std so the
        # parameter can actually move in response to entropy_coef this time.
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
      # 0.005 (was 0.01, 2026-07-27): with the std_range cap (0.27) re-added,
      # Mean action std sat pinned dead flat at the cap from iteration 0 --
      # entropy pressure still pushing max exploration at all times instead
      # of shrinking as the policy converges. Same pairing as the repo's own
      # 2026-07-16 fix for this exact failure mode (commit 4b9018ee): cap
      # alone bounds the blowup, entropy_coef cut is what actually lets std
      # come down. Suspected cause of the stiff-knee "penguin" gait -- noise
      # this large, every step, for the whole run, drowns out fine knee-bend
      # coordination in favor of a noise-robust stiff-legged waddle.
      entropy_coef=0.005,
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
      torque_guidance_coef=0.0,
      torque_guidance_warmup_updates=1000,
    ),
    experiment_name="rhps1_velocity",
    save_interval=150,
    num_steps_per_env=48,
    max_iterations=15_000,
  )
