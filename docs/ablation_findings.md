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

`stance_contacts_mean` matches to three decimals. The whole of v1's divergence
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
