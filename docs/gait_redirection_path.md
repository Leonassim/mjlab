# Redirecting the gait: the path that worked

From the 2.9 cm shuffle that walked on the robot to a 9 cm, 0.79 s stride, in
about 30 hours of fine-tuning on 2026-08-21/23. Every step resumed from the
previous checkpoint with observations and actions untouched, so the weights
always loaded and nothing ever had to relearn walking.

This is written to be replayed as a **curriculum inside a single run**. The
order matters and the reasons are given; the measured effect of each rung is
what justifies keeping it.

## Baseline

`lift`, run `2026-08-21_11-47-31` model_2100 — the checkpoint that walked on
the real robot.

| | |
|---|---|
| step length | 2.9 cm |
| step period | 0.148 s |
| step rate | 6.8 /s |
| sole lift p90 | ~0.8 cm |
| torque mean | 0.376 (policy 0: 0.390) |
| clipping | ~0.10 |
| flat stance | 2.98 of 4 corners |
| falls | 0.5% |

## The rungs, in order

### 1. `freevel` — stop paying for a constant velocity vector

`track_linear_velocity` scored `exp(-|c-v|²/std²)` on **instantaneous**
velocity at std 0.20, weight 3.5 — the largest positive term. A real step is
oscillatory in every component: it decelerates at heel strike, accelerates at
push-off, the CoM rises and falls, the weight swings onto the stance hip. Under
a kernel that narrow all of it reads as tracking error.

Measured: a plausible step profile (±0.35 m/s swing at 0.3 m/s mean) keeps
**24%** of the term against 100% for a constant vector. The shuffle is not what
the policy settled for — it is what the kernel selects.

`direction_progress` low-passes velocity over a gait cycle, then scores it
anisotropically: saturating ramp along the command, wide kernel across it,
nothing on vertical. Overshoot is never punished. 70% of the old weight moves
there; the rest stays at a widened std so the speed command keeps some
authority.

### 2. `freeroll` — split roll/pitch off yaw

`track_angular_velocity` put roll/pitch under the same 0.35 std as the tracked
yaw. For a walking humanoid roll *is* the lateral weight transfer: a 0.3 rad/s
excursion cost ~52% of a weight-3.5 term. `std_xy = 0.8` frees it; yaw keeps
0.35.

### 3. `steplen` — pay per step, not per second

Velocity tracking cannot separate a short fast step from a long slow one: both
average the commanded speed. Rewarding CoM displacement does not separate them
either — displacement per unit time is the velocity again.

`com_step_progress` pays at touchdown, squared, on CoM displacement projected
on the command **and** on the step period, averaged. Two half-steps collect
`2×(0.5)² = 0.5` against `1.0` for one full step over the same ground.

Three implementation details it does not work without:

- **Divide by `step_dt`.** `RewardManager` returns `raw × weight × dt`, so every
  term is a per-second rate. An impulse paid once per touchdown takes that `dt`
  too, which divided this term by ~50 and left it at **0.1%** of the positive
  budget — inert.
- **Debounce the landing.** `compute_first_contact` fires on contact chatter:
  `step_rate` read 9.06/s where the gait produces ~4.8. Every footfall counted
  twice, so the reward was paid twice *and* the accumulator reset mid-step,
  halving the step length it reported. Gate on `last_air_time > 0.05 s`.
- **Size the ceiling, not the starting pull.** `weight × step_rate` is what the
  term is worth once the target is reached, and that is what keeps pulling as
  the policy improves. The run that drove torque to 0.47 started at a perfectly
  ordinary 0.79 but had a 3.6/s ceiling — 37% of the budget, still pulling long
  after the gait had stopped being feasible. Keep the ceiling near 18%.

Targets are walked by a ladder, `0.05 m / 0.25 s` → `0.16 m / 0.80 s`.

### 4. `dense` — pay air time and foot height continuously

`air_time`'s `threshold_max` was 0.25 s against a flight already at ~0.20: the
bonus capped almost where the gait sat, so most of what was being asked for was
past its ceiling. Raised, and switched to the potential-based dense form — same
total over a swing, credited every step instead of at touchdown.

**Foot height got a positive term for the first time.** `foot_swing_height`,
`min_foot_height` and `foot_clearance` are all *penalties*, against 30, 20 and
150 mm targets while the measured peak was ~29 mm p90. Nothing in the budget
had ever paid for lifting. `swing_foot_height_bonus` is linear, not squared —
the squares elsewhere break a tie between gaits achieving the same total, and
there is no tie here, only a quantity that has to move from far below target.
It must be **floor-relative**: `site_pos_w` z sits ~20 mm up with the foot
down, so scoring it raw pays a large constant for standing still.

This is the rung that moved foot height. It went 2.4 → 6.3 cm p90.

### 5. Flat contact

The long stride regressed the support phase: `stance_contacts_mean` 3.05
(policy 0) → 2.98 (lift) → 2.42. The foot lands nearly flat — 2.89 of 4 corners
at touchdown — and rolls onto an edge afterwards. Touchdown is *not* the
problem; support is.

`flat_touchdown` was an impulse crushed by the same `dt` scaling (‑0.012, i.e.
inaudible); made a rate. `flat_support` -2.4 → -4.0. Both corrected in the
right direction within 50 iterations.

### 6. `calm` — quiet the arms and head

Arms clipped 23.2% of joint-samples at 0.512 mean torque, head 17.2% at 0.513 —
the head worked as hard as the arms — against legs at 14.9% and 0.394, which is
policy 0's level exactly. **All the excess torque was above the waist.**

`upper_body_action_acc_l2` was supposed to cover this and did not: its joint set
held all twelve leg joints and six arm joints, with no head, no chest, no
shoulder P/R and no left wrist. It mostly penalised the legs while the two
worst-clipping joints in the robot (L_WRIST_R/Y, ~42%) paid nothing. Rescope it
to the real 18 upper-body joints. Give the head its own weight rather than
averaging it into eighteen.

### 7. Charge for torque, not for movement

The most instructive failure. Penalising upper-body *motion* worked exactly as
written — action acceleration fell 45× — and **arm clipping rose anyway**, 0.239
→ 0.288, with the legs following. Moving less, the policy held the arms in
strained static postures, which clip just as hard.

`torque_limit_margin` is the only term that prices the zone clipping actually
comes from (above `soft_ratio` 0.8) and it was worth -0.16. At -0.40, clipping
fell 0.2297 → 0.2109 **and step length rose** 0.063 → 0.090 over the same 1250
iterations. No trade-off appeared.

### 8. Anti-hover

`air_time_mean` reached 0.759 s against a 0.793 s step period — the foot
airborne essentially the whole cycle, with almost no double support. Visible in
the video as lift, wait, put down.

Three settings allowed it together: `touchdown_cost` zeroed, `threshold_max`
raised to 0.60, and the `overflow_threshold` guard left at 2.0 where it could
never fire. Past 0.60 s nothing was paid *and nothing was charged*.

`threshold_max` 0.40, `overflow_threshold` 0.45. **Do not reinstate
`touchdown_cost` for this**: it charges for landing *often*, which is the
opposite lever. What has to be expensive is staying up.

## Result

| | baseline | now |
|---|---|---|
| step length | 2.9 cm | 9.1 cm |
| step period | 0.148 s | 0.793 s |
| step rate | 6.8 /s | 1.23 /s |
| sole lift p90 | ~0.8 cm | 6.3 cm |
| pre-contact vel | 0.178 (lift) | 0.134 |
| torque mean | 0.376 | 0.467 |
| clipping | ~0.10 | 0.21 |
| flat stance | 2.98 | 2.60 |
| falls | 0.5% | 8% |

Impact velocity is **softer** than both policy 0 and lift despite the foot
falling from six times higher.

## Open defects

1. **`error_vel_xy` 0.53.** The counterpart of `freevel`: an operator has little
   authority over speed. `track_linear_velocity` has to come back up once the
   slow regime is consolidated.
2. **Flat stance 2.60 against policy 0's 3.05.**
3. **Falls 8% against 0.5%.**
4. **The policy now depends on the QP.** `velocity_damper` is fully ramped, so
   the projection is part of the plant it learned against. It is not expected to
   stand without it.

## Traps that cost time

- **`Metrics/peak_height_mean` reads ~70× low.** It resets its peak on the first
  corner touching while landing fires on all four. `log_sole_height` was written
  to replace it. Use `sole_height_p90`. Reported as "foot height" for a full day
  before this was caught.
- **`common_step_counter` is restored from the checkpoint *after* the
  curriculum's first call**, so a relative base captured there reads 0 against a
  counter already at 417820, and every stage fires at once. Counting calls
  instead is worse: the manager calls curriculum terms ~25× per iteration.
  Re-baseline on the counter's restore jump.
- **A ladder must advance on achievement, not elapsed time.** The height ladder
  walked its target to 30 mm on schedule while the foot did not follow, ending
  as the constant it was written to avoid.
- **Rescoping a term changes its raw by orders of magnitude.** `upper_body_
  action_acc_l2` came out at -578 against a 7.6 negative budget after a joint-set
  change carried the old weight across; `head_vel_l2` at -0.035 for the mirror
  reason. Always re-measure the realized `Episode_Reward` and resize.
- **`--video-interval` is in environment steps, ~50 per iteration.** 250 gives a
  video every 5 iterations, not every 250.

## What this path does not establish

Every rung was validated only as a resume from the previous one. No run has ever
been made from scratch with the final configuration, and several rungs were
changed while others were still settling, so attribution is bundled. Before any
real-robot session this sequence should be replayed as a curriculum in one
clean run — which is the purpose of this document.
