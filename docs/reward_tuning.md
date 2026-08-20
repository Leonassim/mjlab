# Testing a corrected reward, and finding the weight that goes with it

Several foot rewards were corrected since policy 0 — the corner count, the
landing bonus, the single-support grace. A correction changes *what is measured*,
so the coefficient in front of it no longer means what it did. Testing the
correction at the old weight, or at a weight guessed to compensate, conflates two
questions. This separates them.

## The trap

`Episode_Reward/<term>` is `weight × cost`. When a correction rescales `cost`,
keeping `weight` changes the term's influence on the objective even though
nothing about the intent changed. `flat_support`'s corner tolerance is the clear
case: counting corners inside a 1 mm band instead of by solver detection makes
the deficit smaller almost everywhere, so at the old −2.4 the term quietly
stopped mattering.

Reading the ratio off two *training runs* does not settle it either — the two
runs learned different gaits, so measurement and behaviour move together. The
factor has to come from the same trajectories.

## Phase 1 — paired calibration, no training

`scripts/tools/reward_scale_factor.py` rolls out one checkpoint and evaluates
each corrected reward under **both** parameterisations on the same states. It
prints, per term, the raw-cost ratio `corrected / July` and the weight that
restores the realized value:

    parity_weight = old_weight / ratio

Ten minutes, no GPU contention worth caring about. Rollouts read absolute
magnitudes low compared with the training loop, but this is a *ratio* between two
variants on identical trajectories, so that bias cancels.

## Phase 2 — one run at parity

`RHPS1_ABLATION=p0+rand+<term> RHPS1_W_<term>=<parity_weight>`, 700 iterations,
against the `+rand` baseline. Judge on all four criteria, not walking alone:
`track_linear_velocity`, `torque_limit_ratio_mean`, `peak_height_mean`,
`fell_down`.

Parity is the null hypothesis — "the correction is a better measurement of the
same intent". If it walks as well as `+rand`, the correction is free and the
question becomes whether to spend the improvement.

## Phase 3 — sweep, only where Phase 2 is ambiguous

Two more runs at `0.5 ×` and `2 ×` parity. Three points on a log scale is enough
to see whether the term is at a plateau or on a slope; a finer sweep is only
worth it once something depends on the difference.

## What the night measured

Realized value per step at iteration 700, `+rand` (July measurement) against
`+feet` (corrected, current weights):

| term | +rand | +feet | shift |
|---|---|---|---|
| `min_foot_height` | −0.303 | absent | **+0.303** |
| `air_time` | +0.030 | +0.191 | **+0.161** |
| `impact_vel` | −0.293 | −0.216 | +0.077 |
| `flat_support` | −0.524 | −0.466 | +0.058 |
| `standing_single_support` | −0.065 | −0.011 | +0.054 |
| `flat_touchdown` | −0.032 | absent | +0.032 |

The block lightens the foot objective by about **+0.68 per step**, and the two
large movers are dropping `min_foot_height` and the landing bonus.

**But that table is confounded** — the two runs learned different gaits, so it
cannot separate a change in measurement from a change in behaviour. Reading it, I
concluded `flat_support` was already near parity. Phase 1 on identical
trajectories says the opposite:

| term | ratio corr./July | weight now | parity | off by |
|---|---|---|---|---|
| `flat_support` | 0.889 | −11.00 | **−2.70** | 4.1× too strong |
| `impact_vel` | 0.445 | −2.00 | −1.12 | 1.8× too strong |
| `air_time` | 1.216 | +2.00 | +1.64 | 1.2× too strong |
| `standing_single_support` | 0.197 | −6.00 | **−20.31** | 3.4× too **weak** |

`flat_support` runs at 4.1× parity: the corrected corner count is only 11%
cheaper on a walking trajectory, so raising −2.4 to −11 overshot badly. Holding
four corners down is worth four times what it should be — a concrete mechanism
for the feet block planting both feet, and confirmed by the `fs` rung alone
reproducing the block's signature to 0.2%.

`standing_single_support` goes the other way, and it is the one that took two
wrong readings to find. `grace_period` makes the penalty **5× cheaper**; the
weight moving −4 → −6 recovers only a third of that, so the term ends up 3.4×
weaker than July. Nothing much now discourages standing on one foot.

### A trap in paired calibration: shared state

The first two runs of this tool reported `standing_single_support` ratio
**exactly 1.000**, twice, under different rollout settings. That looked like a
fact about the reward. It was a bug in the tool: `RewardTermCfg.func` holds the
*instance* the reward manager built, not the class, so `isinstance(func, type)`
is False and both "variants" reused the same object — one `grace_left`, one
`prev_count`. A correction that acts through state was measuring itself.

**Give each variant its own instance** (`type(func)(cfg, env)`), and check that a
stateful correction actually moves the ratio. A perfect 1.000 is a red flag, not
a result.

Ordered by shift, so by expected effect:

1. `mfh` — is `min_foot_height` load-bearing after all? It was called a trap
   (zero for standing, a penalty for a short lift), yet the block that drops it
   lifts 65% less. Run it kept and dropped; if kept wins, find its weight too.
2. `air` — `power` 2 → 1 with `touchdown_cost` 0.15 → 0 makes a linear payout
   with no cost for touching down. A short hop becomes profitable, which is the
   mechanism behind "the foot lifts and never lands".
3. `sss` — 1.5 s of grace plus −6 leaves a realized −0.011 against −0.065. The
   penalty all but disappeared; parity would put it near −36, which is likely
   too blunt and is exactly why the sweep exists.
4. `imp`, `fs` — smaller realized shifts, but `fs` is now first in the queue
   anyway: it reproduces the whole block signature on its own.
5. `sss` at **−20.31** — added after the calibration bug was fixed. The only term
   that is too weak rather than too strong.

## Runnable now

    # calibration
    uv run python scripts/tools/reward_scale_factor.py <checkpoint>

    # one corrected reward at parity
    RHPS1_ABLATION=p0+rand+air RHPS1_W_air_time=<w> \
      uv run train Mjlab-Velocity-Flat-RHPS1 --env.scene.num-envs 4096 --video True

Rungs: `fs`, `air`, `mfh`, `sss`, `imp`. Weight override on any reward term:
`RHPS1_W_<term_name>`.
