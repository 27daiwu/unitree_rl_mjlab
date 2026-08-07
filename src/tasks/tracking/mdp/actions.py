from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class ClippedJointPositionAction(JointPositionAction):
  """Joint position action that clips raw policy output before scale/offset."""

  cfg: "ClippedJointPositionActionCfg"

  def process_actions(self, actions: torch.Tensor) -> None:
    if self.cfg.raw_clip is not None:
      actions = torch.clamp(actions, -self.cfg.raw_clip, self.cfg.raw_clip)
    super().process_actions(actions)


@dataclass(kw_only=True)
class ClippedJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action cfg with normalized raw action clipping."""

  raw_clip: float | None = 1.0

  def build(self, env: ManagerBasedRlEnv) -> ClippedJointPositionAction:
    return ClippedJointPositionAction(self, env)
