# Ablation from policy 0 — findings

Results live in `logs/ablation_results.md`, which is gitignored. This keeps what
the runs concluded.

## Rung 0 — `p0`

### v2, run `2026-08-19_21-25-37` — **tracks policy 0. Baseline is sound.**

| iteration | | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| `air_time_mean` | policy 0 | 0.484 | 1.027 | 0.724 | 0.595 |
| | **p0 v2** | 0.496 | 1.049 | 0.857 | 0.691 |
| | p0 v1 | 0.544 | 1.410 | 1.825 | 2.321 |
| `stance_contacts_mean` | policy 0 | 1.953 | 1.892 | 1.968 | 2.100 |
| | **p0 v2** | 1.952 | 1.894 | 1.947 | 2.081 |
| | p0 v1 | 3.152 | 3.018 | 3.019 | 2.953 |

At iteration 700 it lands within 1.5% of policy 0 on the metric that decides:

| it700 | policy 0 | p0 v2 |
|---|---|---|
| `track_linear_velocity` | 2.492 | 2.456 |
| `stance_contacts_mean` | 2.808 | 2.590 |
| `air_time_mean` | 0.134 | 0.174 |
| `torque_limit_ratio_mean` | 0.363 | 0.385 |
| `error_vel_xy` | 0.266 | 0.275 |
| `fell_down` | 0.003 | 0.001 |

Same curriculum dip at iteration 600 and the same recovery. Torque demand runs
~6% above policy 0, which is small but is the property the ladder exists to
protect -- watch it as rungs stack.

**`peak_height_mean` is not comparable across the two trees.** It comes from a
`TerrainHeightSensor` whose ray origin is the `{prefix}_foot` site, and that site
moved from z = -0.08 to -0.10 (the real sole plane) since July. Judge foot lift
between rungs, all on today's code, never against policy 0.

`stance_contacts_mean` matches to three decimals early on. The whole of v1's divergence
was `flat_support.corner_tolerance = 0.001` in the baseline — counting foot
corners by height inside a 1 mm band instead of by the solver's contact
detection. Set to 0.0 and the reward is July's again.

### v1, run `2026-08-19_18-49-41` — **superseded, the verdict below was wrong**

Read it as a record of how the baseline was wrong, not of the code being wrong.
The conclusion it reached — quoted next — does not hold.

Policy 0's exact configuration on today's tree diverges from policy 0 itself,
while `mjlab-p0` (July code, same configuration) tracks it to three decimals.

| iteration | | 100 | 200 | 400 | 600 | 700 |
|---|---|---|---|---|---|---|
| `track_linear_velocity` | policy 0 | 2.300 | 2.501 | 2.763 | 2.378 | 2.489 |
| | July repro | 2.297 | 2.526 | 2.910 | 2.564 | 2.652 |
| | **today, `p0`** | 2.288 | 2.426 | 2.455 | **1.340** | **1.375** |
| `air_time_mean` | policy 0 | 1.013 | 0.595 | 0.213 | 0.146 | 0.134 |
| | July repro | 0.959 | 0.438 | 0.205 | 0.143 | 0.139 |
| | **today, `p0`** | 1.391 | 2.325 | **3.178** | 3.008 | 2.709 |

Policy 0's air time *falls* to 0.13 s as the robot learns to plant its feet.
Today's *rises* to 3.2 s and stays: the foot stops touching down. The drop in
`track_linear_velocity` at iteration 600 is the curriculum firing (step 24000,
48 steps per env per iteration) and appears in all three runs — but only
policy 0 and the July repro recover from it.

### Ruled out, by measurement

- **Actuators.** The 13 groups shared with policy 0 are identical field for
  field, including armature; the two knee groups differ from policy 0's single
  `.*_KNEE_P` group only by the L/R expression. Worth stating because
  `check_ablation_ladder.py` compares lists by length first and reports
  `actuators: 14 -> 15` as one line without descending into it — the fields were
  verified separately.
- **Foot geometry.** All ten compiled foot collision geoms match between the two
  trees: position, size, `margin`, `gap`, `condim`, `priority`.
- **Early dynamics.** At iterations 0-50 `air_time_mean`, `foot_vel_max`,
  `Episode_Reward/air_time` and `Train/mean_reward` agree to three decimals
  across all three runs. Same plant, same starting policy.

### The one thing that differs from iteration 0

`stance_contacts_mean` reads 3.21 where policy 0 reads 2.35 — before any
learning, on identical dynamics. That is a measurement difference, not a
behavioural one, and it means **the `stance <= 3.0` half of the verdict rule is
on a scale that moved**; `track_linear_velocity` is the half that stayed
comparable. Behavioural divergence starts separately, around iteration 100.

### Why v1 was wrong, and what the method missed

The env.yaml diff records only what a `RewardTermCfg` sets explicitly, so a
signature default that changed since July is invisible to it. Five parameters
passed through that hole. Comparing *effective* parameters — explicit config
plus the defaults in play — finds them; that comparison is now the check that
matters, not the yaml diff alone.

`contact_sensor.py` was the leading suspect and is innocent: the sensor change
(upstream `ddc5e853`, accumulating the exact substep dt) never had to be tested.
So was the plant — the actuator and foot-geometry comparisons in the v1 section
still hold and remain useful.

Rungs 1-8 can now be read.

## Complementary studies, after the ladder

The ladder answers "which deviation costs the gait". These answer "what should
the objective actually be", and each needs its own run rather than a rung.

### 1. `flat_support.corner_tolerance` and its weight

Counting foot corners by relative height inside a 1 mm band, rather than by
whether the solver reported a contact, is the better measurement: the sole is
parallel to the ground to within 16 um in the default pose, yet only 2.15 of 4
corners registered — 130 um of toe-versus-heel offset per milliradian of ankle
pitch lifts two clear of the solver threshold, so the old term was measuring
solver luck.

It is neutralised in the `p0` baseline because policy 0 had no such behaviour,
and restored by the `feet` rung. But the tolerance mechanically *shrinks* the
penalty, so the weight carried alongside it no longer means what it did. The
study is a weight sweep at fixed tolerance, judged on
`Metrics/flat_support_contacts_mean` and `Metrics/peak_height_mean` together:
the point is a flat sole at touchdown that still lifts.

Do not compare `Train/mean_reward` across a tolerance change — the two runs are
on different reward scales from iteration 0.

### 2. Two real-robot defects with no metric

Measured on hardware, both currently invisible to every logged quantity:

- the robot stands too far back on its heels
- it cannot walk backwards without falling immediately

`Metrics/twist/error_vel_xy` aggregates over all directions and hides both. What
is needed:

- **per-direction tracking error**, at least `vx < 0` split from `vx > 0`, so a
  policy that only walks forward stops reading as a policy that tracks the
  command
- **a support/posture metric** — base pitch, or the centre of pressure's position
  along the foot, so "on the heels" becomes a number

Add them as metrics terms (inert, logging only), then re-read the surviving
checkpoints. Adding them mid-ladder would change what every rung is compared
against, so they wait.

### 3. Feasibility without the QP

Policy 0's commands were executable by the robot *without* the QP, and that is
the property the ladder is protecting, not just walking. `torque_limit_ratio_mean`
and `_max` are recorded for every rung. A rung that buys motion by spending
torque headroom is a regression even if it walks better.

## Rung 1 — `+rand`, run `2026-08-19_22-31-51` — **keep**

Better on every criterion, at iteration 700 against the `p0` v2 baseline:

| | p0 v2 | +rand | |
|---|---|---|---|
| `track_linear_velocity` | 2.454 | 2.675 | +9.0% |
| `torque_limit_ratio_mean` | 0.385 | 0.356 | **−7.6%** |
| `sole_height_p90` | 0.0071 | 0.0078 | +10.0% |
| `peak_height_mean` | 0.0015 | 0.0019 | +27.9% |
| `progress_ratio` | 0.542 | 0.628 | +15.9% |
| `error_vel_xy` | 0.276 | 0.227 | −17.7% |
| `air_time_mean` | 0.174 | 0.128 | −26.8% |
| `fell_down` | 0.0009 | 0.0044 | +363% |

Randomisation does not spend torque headroom, it returns some: demand falls 7.6%
while speed rises 9%. The foot also lifts higher for less time in the air — a
firmer step, not a dragging one. `fell_down` triples but from 0.0009 to 0.0044,
which is inside the range this quantity swings on its own; watch it, do not
conclude from it.

## Rung 2 — `+obs`, run `2026-08-19_23-39-10` — **removed from the ladder**

| iteration | 100 | 200 | 400 | 700 |
|---|---|---|---|---|
| `air_time_mean` | 1.375 | 1.868 | 3.186 | 3.077 |
| `progress_ratio` | 0.013 | 0.011 | 0.006 | 0.005 |
| `track_linear_velocity` | 2.289 | 2.378 | 2.438 | 1.344 |

The robot holds a foot up and stops advancing — `progress_ratio` two orders of
magnitude below `+rand`'s 0.63, with no falls. This is the same signature as the
v1 failure, reached by a different route.

Two reasons it is pulled rather than judged:

1. **The rung bundles three changes** — `last_action` → `executed_action`,
   history 0 → 5, and two new critic terms — so a verdict cannot be attributed.
2. **`executed_action` is being tested where it was never validated.** It exists
   to feed back what the torque projection did, and `torque_feasibility_ratio` is
   `None` in both the `p0` rung *and* today's production config. Its own
   docstring promises a fallback to the raw action for actuators without the
   projection, but `_executed_position_target` is allocated unconditionally
   (`finite_difference_pd_actuator.py:249`), so the fallback never fires.
   In July it was validated *together with* the projection at ratio 1.0, and the
   note from then is explicit: "these two are one test, not two".

Precedent also says not to call this one at 700: the July version read
`swing_height` −93% at iteration 500 and had fully recovered by 1000.

Carrying `obs` through the six remaining rungs would contaminate every verdict,
so the chain was restarted on the `+rand` base. `obs` gets its own run
afterwards, split into its three parts and long enough to see a recovery.

## Rung 3 — `+rand+knee`, run `2026-08-20_00-52-02` — **breaks walking**

Knee effort limit 100 → 70 N·m, with the action scale from 0.0075 to 0.00525.

| it700 | +rand | +rand+knee |
|---|---|---|
| `progress_speed` (commanded 0.232) | 0.149 | **0.006** |
| `left_foot_marker_speed` | 0.174 | 0.025 |
| `right_foot_marker_speed` | 0.173 | 0.024 |
| `flat_support_contacts_mean` | 2.586 | 1.604 |
| `track_linear_velocity` | 2.674 | 1.332 |
| `sole_height_p90` | 0.0078 | 0.0091 |

The video settles it: legs straight and held together, robot planted on the
front of its feet, not walking. `flat_support_contacts_mean` at 1.60 of 4 says
only the toe corners touch. Both feet are symmetric and 7x slower than `+rand`,
so this is not a foot parked in the air — it is a statue on tiptoe, the failure
mode `ankle_pitch_torque` was added to discourage.

This reproduces the July finding that the knee ceiling costs translation, this
time cleanly. **It is a hardware constraint, not a rung to accept or reject**:
the real knee cannot deliver 100 N·m, so the objective has to be adapted to 70,
not the other way round. Left out of the remaining chain for the same reason as
`obs` -- stacking on a base that does not walk measures nothing.

Note the rung bundles two changes, effort limit and action scale, so which of
the two costs the gait is still open.

**Resolved — `air_time_mean` is a conditional mean, and I was reading it wrong.**
`rewards.py:534` divides by the number of feet *currently airborne*, not by the
number of feet:

    foot_air_time = max(split_air, dim=2) * foot_in_air
    mean_air_time = sum(foot_air_time) / clamp(sum(foot_in_air), min=1.0)

So 3.45 s does not mean "feet spend 3.45 s in the air". It means "on the rare
occasions a foot is airborne, it stays up 3.45 s". That is fully consistent with
both feet planted and nearly motionless: the robot almost never lifts, and when
it does, the foot does not come back down. No contradiction, and the reading is a
genuine alarm — **a foot that leaves the ground and does not land**.

This also explains why `+rand+feet` shows a normal 0.134: there the feet do land,
every stride, they simply do not lift high. Two different failures, and the metric
separates them correctly.

## Ladder structure, revised

Cumulative stacking assumed each rung would survive. Two do not, and each one
that fails invalidates everything above it. The chain now runs the remaining
rungs on the `+rand` base:

    p0+rand+feet → +prox → +pose → +mirror → +static

`obs` and `knee` become their own studies. The consequence is that the final
rung no longer reconstructs today's configuration, so the third assertion in
`check_ablation_ladder.py` is now a statement about the *definitions*, not about
what was run.

## Rung 4 — `+rand+feet`, run `2026-08-20_02-01-05` — **this is the current config's failure**

| it700 | +rand | +rand+feet | |
|---|---|---|---|
| `stance_contacts_mean` | 2.577 | **3.793** | +47% |
| `flat_support_contacts_mean` | 2.585 | 3.833 | +48% |
| `sole_height_p90` | 0.0078 | 0.0035 | −55% |
| `peak_height_mean` | 0.0019 | 0.0007 | −65% |
| `progress_speed` | 0.149 | 0.049 | −67% |
| `track_linear_velocity` | 2.675 | 1.728 | −35% |
| `foot_vel_max` | 1.045 | 0.737 | −30% |
| `air_time_mean` | 0.128 | 0.134 | normal |
| `fell_down` | 0.0044 | 0.0000 | never falls |

A different failure from `knee`'s: the feet stay **flat on the ground**, all four
corners, and the robot shuffles without lifting. `stance_contacts_mean` 3.79 of 4
is the exact signature recorded at the start of this investigation for the runs
that would not walk ("1.95 and 3.6-3.8"). One block of the config reproduces it.

`air_time_mean` is normal here, which separates this failure from the tiptoe one
and confirms the 3.2-3.5 s reading is specific to that other mode.

**The mechanism is the one Léo predicted.** `corner_tolerance = 0.001` makes the
corner count generous, which mechanically shrinks the `flat_support` penalty, so
the weight was raised −2.4 → −11 to compensate. Together they overshoot: keeping
four corners down becomes worth more than stepping, and the policy stops lifting.
The tolerance is the better measurement; the weight that belongs with it has not
been found yet. That pairing is the first complementary study, and it now has a
measured failure to calibrate against.

The block also carries seven other changes (foot_clearance −4 → −10, impact_vel
−0.5 → −2, standing_single_support −4 → −6, foot_slip −0.3 → −0.5, air_time
power 2 → 1 and touchdown_cost 0.15 → 0, and dropping `min_foot_height` and
`flat_touchdown`), so `flat_support` is the leading suspect, not a proven one.
Splitting it is the natural follow-up.

## Method correction: cumulative was the wrong design here

Three of the first four rungs fail, and each failure invalidates everything
stacked on top — three restarts in one night. The chain now runs **one rung at a
time on the `+rand` base**, so every run differs from a walking configuration by
exactly one block:

    p0+rand+prox, p0+rand+pose, p0+rand+mirror, p0+rand+static

Cumulative made sense when the expectation was that most rungs would survive and
the ladder would end on today's configuration. With a majority failing, clean
attribution is worth more than reaching the endpoint.

## Rungs 5-8, isolated on `+rand`

All at iteration 700, each differing from `+rand` by one block.

| | `+rand` | `+prox` | `+pose` | `+mirror` | `+static` |
|---|---|---|---|---|---|
| `track_linear_velocity` | 2.675 | 2.474 | 2.727 | 2.539 | 2.032 |
| `progress_speed` | 0.149 | 0.128 | 0.156 | 0.134 | **0.025** |
| `torque_limit_ratio_mean` | 0.356 | 0.345 | **0.412** | 0.346 | **0.473** |
| `peak_height_mean` | 0.0019 | 0.0014 | 0.0013 | 0.0016 | 0.0007 |
| `stance_contacts_mean` | 2.577 | 3.077 | 2.311 | 2.780 | 1.796 |
| `fell_down` | 0.0044 | 0.0104 | 0.0141 | 0.0049 | **0.0931** |

**`+prox` — a trade, not a defect.** Walks, but lifts a quarter less and drifts
toward flat feet (3.08). Hardware protection paid for in stride amplitude; torque
is slightly better. A decision to take, not a bug to fix.

**`+pose` — reject.** +1.9% speed for **+15.8% torque demand**. On walking alone
it would have been kept; on the feasibility criterion it fails. The foot also
lifts 35% less while air time doubles — a softer step, not a longer one. The
`pose` reward itself is unchanged (0.484 → 0.486), so the policy pays the same
price for a looser posture.

**`+mirror` — insurance against a risk that is not present here.** It works:
foot-speed asymmetry 0.51% → 0.03%, seventeen times more symmetric. But 0.51% was
already symmetric, and the mirror loss was introduced against one-leg gaits with
foot-speed ratios of 1.5-1.8; this is 1.005. Costs 10% progress and 17% foot
lift. Leave it off for now, keep it available — not a rejection like `pose`.

**`+static` — the worst rung, and the only one that makes the robot fall.**
Progress down 83%, torque demand up 33%, falls up **21x** to 9.3%. It drops the
four curricula and holds `rel_standing_envs` at 0.4 where policy 0 fell to 0.1 —
four times as many environments told to stand still — and gives `air_time` its
full weight of 5.0 from step 0 instead of ramping 2 → 5. `air_time_mean` 1.08
puts it in the "foot lifts and does not land" mode: the landing bonus is worth
too much too early, so the foot goes up to collect it.

## Summary

One rung of eight improves without a counterpart. Four break walking. The current
configuration did not drift by accumulating small costs — it carries two blocks
that break it outright (`feet`, `static`), one hardware constraint that breaks it
(`knee`), and a set of changes ranging from neutral to unfavourable.
