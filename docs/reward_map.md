# What every reward term actually does

    uv run python scripts/tools/reward_map.py <run> [<run>...] --it 700

Weights say what a term is worth in principle. `reward_map.py` reads
`Episode_Reward/*` off the TensorBoard logs and says what it was worth in fact:
realized value per step, share of its own sign's budget, and `cost = value /
weight` -- the raw measurement, which is where saturation shows.

Measured on `p0+rand` at iteration 700, cross-checked against policy 0.

## The shape of the objective

30 terms. 5 are exactly zero on every policy measured, 5 more are under 1%, and
two of the three large bonuses sit at their ceiling:

| term | cost | of max |
|---|---|---|
| `upright` | 0.985 | 98.5% |
| `pose` | 0.968 | 96.8% |
| `track_angular_velocity` | 0.902 | 90.2% |
| `track_linear_velocity` | 0.764 | 76.4% |

A saturated bonus is a constant: it holds budget without carrying gradient.
`upright` and `pose` together are 37% of the positive budget and read the same
on a walking policy and on a both-feet-planted one.

Penalties split **65% regularisation / 34% foot shaping**. The regularisation
block -- `torque_limit_margin` above all -- is what makes the actions executable
without the QP. Anything done to the foot block must not dilute it.

## Three foot terms measure something other than their name

`air_time` pays `(min(t_air, threshold_max)/threshold_max)**2 - 0.15` once at
landing, and **the curriculum moves the target away from the policy**:
`threshold_max` steps 0.1 -> 0.3 -> 0.5, so break-even
(`t = threshold_max * sqrt(0.15)`) walks 0.039 -> 0.116 -> 0.194 s while nothing
makes the gait's flight time follow.

Its effective weight is **5.0**, not the 2.0 in `env.yaml` -- a second curriculum
term raises it. Read the weight from `Curriculum/air_time_weight`, never from the
dump. At that weight the term still returns 0.3% of the positive budget, against
9.7% for `foot_swing_height`, which is also charged per landing: a ~30x
difference in what one landing is worth.

Whether the gait sits above or below break-even is **not settled** by
`Metrics/air_time_mean` -- that is a conditional mean over currently-airborne
feet, not the flight duration at touchdown. `reward_audit.py` measures the
landing distribution directly.

`min_foot_height` charges `max(0.08 - z, 0)` on every airborne step. Foot down,
zero. Foot up 2 cm, the gate flips and the penalty **jumps 0 -> -0.30**, and only
clears past 8 cm. Attempting a step costs more than standing still: a cliff, not
a slope. Raising the weight deepens it, which is why `peak_height` never
responded to it.

`foot_clearance` charges `|z - 0.15| * ||v_xy||` on absolute height. The foot is
never near 0.15, so the first factor is a near-constant 0.10 and what is actually
being taxed is **swinging the foot forward**. Weight -4.0.

`foot_swing_height` is the one with the right shape -- peak per swing, charged
once at landing -- but its 0.15 m target is two orders above where the policy
operates, so it sits at 97% of its maximum cost permanently. It is a per-landing
tax, and the only way to lower it is to land less.

## Two degenerate exits

The two observed failure modes are the two ways to satisfy the foot block without
walking. Costs (value / weight), iteration 700:

| exit | term | walking | broken | x cheaper |
|---|---|---|---|---|
| plant both feet (`+fs`) | `flat_support` | 0.218 | 0.041 | 5.4 |
| never land (`+fs@-3.24`) | `min_foot_height` | 0.061 | 0.004 | 13.6 |
| | `impact_vel` | 0.586 | 0.062 | 9.4 |
| | `foot_clearance` | 0.041 | 0.005 | 7.9 |

Exit B zeroes three terms at once -- 13.5% of the penalty budget -- because none
of them can be charged if the foot never comes down.

## flat_support: the whole term is gated on `loaded`

`cost = sum(deficit^2 * charge_mask)` and, in July's parameterisation,
`charge_mask = loaded`. An airborne foot carries no force, is not loaded, and is
not charged at all. **Lifting a foot exits the penalty**, so any change that
raises the price of standing on it pushes the policy off the ground. Split three
ways at July's own weight of -2.4, two of the three do exactly that:

| rung | change | verdict | air_time_mean |
|---|---|---|---|
| `fsct` | `corner_tolerance` 0 -> 0.001 | broken, hovers | 3.26 |
| `fscg` | `change_gain` 0 -> 1.0 | **walks** | 0.118 |
| `fsload` | `load_threshold` 0 -> 140 N, `standing_threshold` -1 -> 0.1 | broken, hovers | 4.39 |

Different routes to the same exit:

- `fsct` stops counting corners by solver contact and starts counting them
  within 1 mm of the foot's own lowest corner -- a flatness measure, not a
  contact one. Across a ~25 cm sole that is 4 mrad of ankle pitch, held for the
  whole of stance, while a real gait rolls heel to toe. Being loaded got much
  more expensive.
- `fsload` charges nothing below 140 N. That is a threshold to duck under: keep
  the foot grazing and the penalty is gone. It closes the standing dodge
  (`standing_threshold` charges an unloaded foot the full deficit at zero
  command) while opening a new one for every walking env.
- `fscg` charges only *changes* in corner count on feet loaded in both steps.
  No new escape route, and it telescopes to zero over a clean stance -- the only
  one of the three that leaves walking intact.

This is why `fs` failed at -11 and at -3.24 by opposite modes. It was never a
dosage question.

## Consequences for the ladder

- Do not tune weights on a term whose shape is wrong. `fs` failed at both -11 and
  -3.24 by opposite modes; that is a shape problem, not a dosage problem.
- The height rewrite is: drop `min_foot_height` (the cliff), bring
  `foot_swing_height`'s target down to something reachable, and raise its weight
  to carry the pressure alone. The `mfh` rung did only the first half, which is
  why it cost 20% of foot height.
- `track_angular_velocity` outpays `track_linear_velocity` at equal weight and
  has less headroom left. The margin is in advancing, not yawing.

Related: [reward_tuning.md](reward_tuning.md) for the paired-calibration
protocol, [ablation_findings.md](ablation_findings.md) for the rung verdicts.
