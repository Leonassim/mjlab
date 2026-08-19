# Ablation from policy 0 — findings

Results live in `logs/ablation_results.md`, which is gitignored. This keeps what
the runs concluded.

## Rung 0 — `p0`, 2026-08-19, run `2026-08-19_18-49-41`

**Verdict: broken. The regression is in the code, not the configuration.**

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

### Next test

Net diff on `src/mjlab/sensor/` between the two trees is one file, 19 lines:
`contact_sensor.py`, where air time changes from differencing the float32 sim
clock to accumulating the exact substep dt (upstream `ddc5e853`). Port that file
alone into `mjlab-p0` and run ~400 iterations:

- air time climbs toward 3 s → the contact sensor is the cause, one file
- air time stays near 0.21 → the sensor is innocent; bisect the nine rewritten
  reward functions instead

Until this is resolved, rungs 1-8 are built on a baseline that does not walk and
measure nothing.
