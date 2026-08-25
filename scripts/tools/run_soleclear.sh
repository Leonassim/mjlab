#!/usr/bin/env bash
# Fix the foot-lift measurement and resume from the checkpoint that showed the
# defect on hardware.
#
# it13200 logged 6.3 cm of foot lift and trips almost immediately on the real
# robot: the foot swings pitched, front edge near the floor. swing_foot_height_
# bonus scored one site at the middle of the sole, so pitching paid for itself.
#
#   soleclear   score min over the sole, box overhang included
#   slowstep    rebase the step-period ladder: 0.58 -> 0.90 s
#
# it13200 measures step_period 0.530 and air_time 0.497 -- the foot is off the
# ground 94% of its own cycle, so there is no double support left to stand on.
# That is the instability seen on hardware. Slowing the step takes the swing
# fraction to 50% without touching the air cap, and makes 3 cm of real
# clearance affordable: lifting that far inside 0.53 s needs vertical velocity
# the torque budget does not have, which is the corner a policy escapes by
# pitching the foot instead of lifting it.
#
# Nothing else changes, so the verdict is attributable. Watch
# Metrics/sole_height_overstated_mean first -- it IS the cheat, in metres, and
# the first iterations measure how much of the 6.3 cm was ever real. Then
# Metrics/sole_clearance_p90 for whether honest clearance actually rises.
#
# Guard rails unchanged: Metrics/torque_saturated_frac and impact velocity must
# not move, and standing_single_support must not climb past it13200's 0.402.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]
do sleep 60; done
sleep 30
export RHPS1_ABLATION=${RHPS1_ABLATION:-p0+rand+lift+steplen+freevel+freeroll+dense+calm+stable+soleclear+slowstep+softland+landtime+groundtax+freearms+impactladder}
# Clearance target just above the 0.0263 the gait actually reaches. At 0.030 the
# term kept a gradient the plant cannot satisfy: over the last 40 iterations
# clearance moved +0.4% and period -1.2% while clipping climbed +1.2%, i.e. the
# demand was being paid for in torque and delivering nothing. Just-above-measured
# is where a target belongs.
export RHPS1_CLEAR_TARGET=${RHPS1_CLEAR_TARGET:-0.027}
# Rebalance freevel toward the command. It moved 90% of track_linear_velocity
# onto direction_progress, whose ramp SATURATES at the commanded speed, and
# widened what remained to a 0.70 kernel -- so speed stopped following the
# command at all. The policy settled at a comfortable 0.159 m/s whatever it was
# asked, and was never trained to be stable faster, which is exactly how it
# behaves on the robot: fine at 0.159, falls above.
#
# 0.6/0.45, not back to the original 0.9/0.20: the narrow kernel is what
# selected the shuffle in the first place, and calls every excursion of a long
# step an error. This keeps the long step and makes the speed command mean
# something again.
export RHPS1_FREEVEL_SHARE=${RHPS1_FREEVEL_SHARE:-0.6}
export RHPS1_FREEVEL_STD=${RHPS1_FREEVEL_STD:-0.45}
export WANDB_INIT_TIMEOUT=300 WANDB__SERVICE_WAIT=300
exec .venv/bin/train Mjlab-Velocity-Flat-RHPS1 \
  --env.scene.num-envs 4096 --video True \
  --video-interval ${VIDEO_INTERVAL:-7200} --video-length ${VIDEO_LENGTH:-600} \
  --agent.resume True \
  --agent.load-run ${LOAD_RUN:-2026-08-25_18-30-00} \
  --agent.load-checkpoint ${LOAD_CKPT:-model_15450.pt}
