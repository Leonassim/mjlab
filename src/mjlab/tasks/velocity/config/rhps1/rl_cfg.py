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
        # 3.0 (was 0.4286, 2026-07-29): leg action scale went 7.0 -> 1.0
        # (rhps1_constants._LEG_SCALE_MULTIPLIER), paired with the actuator's
        # torque_feasibility_ratio. Two independent derivations agree on 3.0,
        # which is why it is not a guess:
        #
        #   - Physical amplitude, the invariant this file has preserved across
        #     every rescale: 1-sigma should be 0.021 rad (~1.2 deg) on the
        #     binding joint. CROTCH_P's scale is now effort_limit/stiffness =
        #     140/20000 = 0.007 rad/unit, so 0.021 / 0.007 = 3.0.
        #   - Mean drift rate, the reason 7.0 was adopted in the first place:
        #     PPO moves the policy mean at a rate proportional to sigma in
        #     raw-action space, while the raw excursion needed for a given
        #     physical excursion goes as 1/scale, so time-to-reach goes as
        #     1/(sigma*scale). Preserving 0.4286*7.0 = 3.0 at scale 1.0 gives
        #     sigma = 3.0, i.e. the same reachability that made 7.0 work.
        #
        # Sanity check against the projection: 3.0 units of noise is 3 * 0.007
        # = 0.021 rad, and the feasibility window at ratio 3.0 is 3 * 0.007 =
        # 0.021 rad. Exploration noise and the feasible set are the same size
        # by construction -- most samples are executable as commanded, and the
        # ones that are not sit right at the boundary where the projection and
        # pd_demand_excess both produce real gradient. That coincidence is the
        # whole design: it is what fails at scale 7.0, where the window is 1/7
        # of a unit against the same noise.
        # 0.05714 = 2 * RHPS1_ACTION_THRESHOLD_LEG, was 1 threshold (2026-07-30).
        #
        # Sizing sigma so the *per-step noise* is torque-feasible was the wrong
        # objective taken alone. What governs learning is the trajectory
        # excursion measured in sigma, and 1 threshold wrecked it: a 20 deg hip
        # swing needs 7.1 action units under either setting, but the old
        # scale-7.0 config had sigma = 1.0 unit, so the mean had to travel
        # 7 sigma -- at 1 threshold it must travel 248. PPO moves the mean about
        # 0.14 sigma per update under desired_kl=0.01, so the same behaviour went
        # from ~50 updates away to ~1700. Same scale, same excursion; only sigma
        # changed, and it made the optimisation 35x harder.
        #
        # 2 thresholds halves that. It does not resolve the underlying tension:
        # feasibility constrains the per-step *change* while learning needs
        # *amplitude*, and iid noise welds the two together, so useful
        # exploration wants 12-50 thresholds where feasible exploration allows 1.
        # Decoupling them needs temporally correlated noise (OU, or noise on a
        # low-frequency basis) instead of rsl_rl's per-step iid Gaussian. That is
        # the real fix and it belongs in rl_ext.py, not in this constant.
        #
        # Cost: per-step noise now demands ~2x the effort limit, so the
        # projection clips more and Metrics/command_infeasible_fraction roughly
        # doubles. Physically harmless -- the projection delivers exactly what
        # the effort clamp would have -- just wasted exploration.
        # 1.0 unit (2026-07-30, was 0.05714 = 2 thresholds).
        #
        # Six reward configurations were tried on 2026-07-30 -- tighter tracking
        # kernel, min_foot_height's landing trap removed, flat_support restored
        # via a contact margin, pre-swing weight transfer, gait_phase stance
        # discounted, a stall termination -- and Metrics/foot_vel_max sat at
        # 0.52-0.55 in every single one. The previous run, the only configuration
        # of this robot that ever produced real foot motion, read 1.79. The one
        # thing those six shared and it did not: sigma_phys of 0.0014-0.0044 rad
        # against its 0.049 rad. Six objectives, one level of foot motion -- the
        # reward is not what caps the movement.
        #
        # So this restores that run's exploration setting exactly (scale is
        # already identical at 0.049 rad/unit) while keeping every verified fix
        # from today. Cost: per-step noise demands ~35x the effort limit, so the
        # projection clips hard and command_infeasible_fraction returns to the
        # 80-90% the old run showed. Physically that is a no-op -- the projection
        # delivers exactly what the effort clamp would -- and deployment is
        # deterministic, so none of it reaches the robot. What it costs is
        # exploration efficiency, which is the thing we were wrong to optimise
        # for: feasibility matters for the *deployed* policy, not for the noise.
        #
        # sigma is learned, so this is also a measurement. Under v8's reward the
        # policy drove it down (0.060 -> 0.050); under v9's it drove it up
        # (0.060 -> 0.090) and settled. If from 1.0 it collapses back toward
        # 0.09, sigma is not the answer and that is worth knowing.
        # 5.0 units at gain 1 (2026-07-30). With kd/dt = 4*kp on the legs, a
        # *sustained* offset of 5 units is exactly e/kp, i.e. one sigma of held
        # noise commands precisely the effort limit -- the natural design point
        # now that the gain no longer hides it, and the reason held noise is
        # what makes this readable: a sustained offset is what the limb actually
        # tracks, where a one-step jump of 1 unit is what saturates instantly.
        # Restored to the 2026-07-29_01-13-36 baseline (2026-07-30): that run
        # is the only configuration of this robot observed to walk. Everything
        # this session did to sigma was chasing a motionless optimum that the
        # sigma reduction had itself created.
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
        # 4.5 (was 1.3, 2026-07-29). A pure rescale of the old cap would give
        # 9.1 (1.3 * 0.049 = 0.064 rad, / 0.007 = 9.1), but the projection
        # changes what a cap is for, so the physical value is deliberately
        # tightened to 4.5 * 0.007 = 0.0315 rad (~1.8 deg).
        #
        # Reason: the projection does not delete the flat region, it *moves*
        # it. Beyond the feasibility window the sampled action and its
        # projection produce identical dynamics and identical advantage, so
        # that noise is still unusable -- it is simply now 1 action unit out
        # (at the final ratio 1.0) instead of 7+. Sigma therefore has to end up
        # near 1.0 for exploration to be spent inside the feasible set, and a
        # 9.1 ceiling would leave room for exactly the runaway that took std
        # 0.43 -> 3.73 to reassert itself over the ramp.
        #
        # Still above init_std (4.5 vs 3.0), which is the non-negotiable part:
        # torch.clamp has exactly zero gradient outside the range in *both*
        # directions, so a cap at or below init_std is not a tight cap, it is a
        # disabled parameter -- the bug that froze std_param at 0.27 for 289
        # iterations. Watch Mean action std against
        # Metrics/torque_saturation_fraction: sigma should fall toward ~1 as
        # the projection tightens, not sit pinned at 4.5.
        # (0.3, 4.0) thresholds. The ceiling had to move with init_std: leaving
        # it at 2.0 while init_std became 2.0 would have put the cap *at* the
        # initial value, and torch.clamp has exactly zero gradient on the
        # boundary in both directions -- the dead-parameter bug this file already
        # records (cap 0.27 under init_std 0.4286 froze std_param for 289
        # iterations). 4.0 keeps 2x of headroom above init.
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
      # 0.004 (was 0.005, 2026-07-29). Note first what did *not* justify a
      # change: the entropy bonus is coef * sum(log sigma + const), so its
      # gradient on log sigma is exactly coef regardless of action scale --
      # the 7.0 -> 1.0 rescale is scale-invariant here and on its own would
      # mean leaving this alone (unlike init_std and std_range, which are
      # amplitudes and had to be restated).
      #
      # The small cut is for the projection, not the rescale. Previously the
      # saturated region was a flat plateau in the return, so entropy there was
      # free: extra sigma cost nothing and bought the bonus, which is how std
      # ran 0.43 -> 3.73 monotonically over 2876 iterations. With the target now
      # projected onto the feasible set, noise past the window is discarded
      # before it reaches the plant -- so large sigma still collects the bonus
      # but no longer buys any exploration. A slightly weaker bonus keeps the
      # policy from paying for noise it cannot use, without risking the
      # premature sigma collapse a larger cut would invite (std_range's floor is
      # 1e-6, so there is no safety net on the way down).
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
