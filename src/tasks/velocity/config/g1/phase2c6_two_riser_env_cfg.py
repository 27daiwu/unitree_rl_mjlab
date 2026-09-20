"""Phase 2C.6 two-riser zero-shot audit task; reward contract is inherited."""
from copy import deepcopy
from mjlab.envs import ManagerBasedRlEnvCfg
from .phase2c4_16cm_reward_intervention_env_cfg import unitree_g1_phase2c4_16cm_reward_intervention_env_cfg
from .phase2c6_two_riser_terrain_cfg import PHASE2C6_TWO_RISER_TERRAIN_CFG

def unitree_g1_phase2c6_two_riser_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_phase2c4_16cm_reward_intervention_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(PHASE2C6_TWO_RISER_TERRAIN_CFG)
  cfg.scene.terrain.max_init_terrain_level = None
  # Existing single-riser diagnostics are not the repeated-riser success metric.
  cfg.metrics = {}
  return cfg
