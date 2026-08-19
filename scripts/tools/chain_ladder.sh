#!/usr/bin/env bash
# One rung at a time on the +rand base, no longer cumulative.
#
# Cumulative stacking assumed rungs would survive. Three of four did not, and
# each failure invalidates everything above it -- three restarts in one night.
# With a majority failing, one-at-a-time is the design that actually attributes:
# every run differs from +rand by exactly one block.
#
#   obs   pulled: bundles three changes, and executed_action is fed back without
#         the torque projection it was validated with
#   knee  pulled: 70 N.m gives a statue on tiptoe. A hardware constraint to
#         design around, not a rung to accept or reject
#   feet  pulled: plants both feet flat and stops lifting -- stance_contacts 3.79
#         of 4, foot lift down 65%. This is the current config's own failure
#         signature, reproduced from one block
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+prox \
  p0+rand+pose \
  p0+rand+mirror \
  p0+rand+static
