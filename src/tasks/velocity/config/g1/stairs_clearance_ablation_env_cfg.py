"""Phase 2A.8 isolated foot-clearance weight ablation environment."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .stairs_baseline_env_cfg import unitree_g1_stairs_baseline_env_cfg


def unitree_g1_stairs_clearance_ablation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Clone the Phase 2A.7 baseline and change only one reward weight."""
  cfg = deepcopy(unitree_g1_stairs_baseline_env_cfg(play=play))
  cfg.rewards["foot_clearance"].weight = -2.0
  return cfg
