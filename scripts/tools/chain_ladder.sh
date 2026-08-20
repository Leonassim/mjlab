#!/usr/bin/env bash
# Reward decomposition and weight calibration, on the +rand base.
#
# The feet block broke walking as a whole; these split it and test the weight
# question the correction raises. Paired calibration
# (scripts/tools/reward_scale_factor.py, walking policy, identical trajectories)
# gives the parity weights:
#
#   flat_support  ratio 0.740  production -11.00  parity -3.24   (3.4x over)
#   impact_vel    ratio 0.445  production  -2.00  parity -1.12
#   air_time      ratio 1.269  production  +2.00  parity +1.58
#
# fs runs twice on purpose: at the production weight it isolates flat_support
# from the block's seven other changes, at parity it tests whether the weight
# alone explains the failure.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+fs \
  p0+rand+fs@-3.24 \
  p0+rand+mfh \
  p0+rand+air@1.58 \
  p0+rand+imp@-1.12
