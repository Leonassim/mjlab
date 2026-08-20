# Splitting the observation change

`obs` bundled four changes and broke walking as a block — the foot lifted and
never landed, `progress_ratio` 0.005 against `+rand`'s 0.63. Nothing about that
result says which of the four did it. These run one at a time, cumulatively, so
each run adds exactly one thing to the one before.

## The four, in order

| rung | change | why it is where it is |
|---|---|---|
| `hist` | `actions.history_length` 0 → 5 | Cheapest to reason about: same observation, five frames of it. Observation 126 → 246. |
| `exec` | `last_action` → `executed_action` | Only meaningful on top of history, and only honest with the projection — see below. |
| `ctorque` | critic gains `joint_torques` | Critic-only, so it can only change the value estimate, never the policy directly. |
| `cscan` | critic gains `foot_height_scan` | Same. |

    RHPS1_ABLATION=p0+rand+hist
    RHPS1_ABLATION=p0+rand+hist+exec
    RHPS1_ABLATION=p0+rand+hist+exec+ctorque
    RHPS1_ABLATION=p0+rand+hist+exec+ctorque+cscan

Four runs, ~1 h each, each against `+rand`.

## `exec` needs its projection, and that is a finding on its own

`executed_action` exists to feed back what the torque projection actually
executed, instead of the raw intent. In July it was validated **together with**
`torque_feasibility_ratio = 1.0`, and the note from then is explicit: "these two
are one test, not two."

`torque_feasibility_ratio` is `None` in the `p0` rung **and in today's production
configuration**. So `executed_action` currently reports the filtered position
target rather than a projected one. Its own docstring promises a fallback to the
raw action for actuators without the projection — that fallback never fires,
because `_executed_position_target` is allocated unconditionally
(`finite_difference_pd_actuator.py:249`).

So `exec` is run twice:

    RHPS1_ABLATION=p0+rand+hist+exec          # as production has it today
    RHPS1_ABLATION=p0+rand+hist+exec+proj     # as it was validated

If the second walks and the first does not, the answer is not "drop
`executed_action`" but "it was never supposed to run without the projection",
and today's production config carries that mismatch.

## Do not call these at 700

The July version of this change read `swing_height` −93% and `air_time` −71% at
iteration 500 and had **fully recovered by 1000**. Judging it early would have
thrown away the change that eventually cut falls by 88%.

So for the observation rungs specifically: run to **1500**, and read 700 / 1000 /
1500. A rung still flat at 1500 is a real negative; one recovering between 700
and 1000 is the known behaviour, not a failure. Budget ~2.3 h per run rather
than 1 h.

That makes five runs of ~2.3 h — a full night for the observation question alone.
Worth it: history plus `executed_action` is what the deployed stack needs to see
the actuator's hidden state, and right now nobody knows which of the four costs
the gait.
