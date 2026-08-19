#!/usr/bin/env bash
# Rungs 4-8 on the +rand base. Both obs and knee are out, for the same reason:
# each stops the robot advancing, so stacking on top of one measures nothing.
#
#   obs   bundles three changes and feeds back executed_action without the
#         torque projection it was validated with
#   knee  effort 100 -> 70 N.m: the leg cannot flex enough to walk, so the
#         policy locks it straight and stands on its toes (1.60 of 4 corners
#         in contact, foot speed 7x down, progress 0.006 against 0.232
#         commanded). This is a hardware constraint, not a rung to accept or
#         reject -- the objective has to be adapted to it separately.
set -u
cd /home/lmoussafir/mjlab-rhps1 || exit 1
exec .venv/bin/python scripts/tools/ablation_series.py \
  p0+rand+feet \
  p0+rand+feet+prox \
  p0+rand+feet+prox+pose \
  p0+rand+feet+prox+pose+mirror \
  p0+rand+feet+prox+pose+mirror+static
