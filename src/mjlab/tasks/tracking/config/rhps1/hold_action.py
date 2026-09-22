"""Actions ignored for a while at the start of a standing-start episode.

mc_mujoco holds the robot at q0 for 3 s before arming, so the policy takes over a
SETTLED robot. Training let it act from the first step, on a state written the
same instant. Measured 2026-09-19 on model_2500: 2.07 m in 12 s acting
immediately, 0.05 m after a 3 s settle -- and 0.05 m in mc_mujoco. The start
transient was the trigger, not the command.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


class HoldableJointPositionAction(JointPositionAction):
  def process_actions(self, actions: torch.Tensor) -> None:
    hold = getattr(self._env, "_action_hold_steps", None)
    if hold is not None:
      held = self._env.episode_length_buf < hold
      if bool(held.any()):
        actions = torch.where(held[:, None], torch.zeros_like(actions), actions)
    super().process_actions(actions)


@dataclass(kw_only=True)
class HoldableJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env) -> HoldableJointPositionAction:
    return HoldableJointPositionAction(self, env)
