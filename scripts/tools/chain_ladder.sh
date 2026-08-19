#!/usr/bin/env bash
# Rungs 3-8 on the +rand base, obs left out.
#
# `obs` bundles three changes (executed_action, history 0->5, two critic terms)
# and failed at iteration 700: air_time 3.19, progress_ratio 0.005 -- the robot
# holds a foot up and stops advancing. It is also being tested in a
# configuration it was never validated in: executed_action exists to report what
# the torque projection did, and torque_feasibility_ratio is None here. Carrying
# it through six more rungs would contaminate every verdict, so it comes out and
# gets its own longer run afterwards.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+knee \
  p0+rand+knee+feet \
  p0+rand+knee+feet+prox \
  p0+rand+knee+feet+prox+pose \
  p0+rand+knee+feet+prox+pose+mirror \
  p0+rand+knee+feet+prox+pose+mirror+static
