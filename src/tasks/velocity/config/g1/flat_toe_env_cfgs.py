"""Explicit Flat toe-clearance reward experiments."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.velocity.mdp.rewards import flat_toe_clearance

from .env_cfgs import unitree_g1_flat_env_cfg


def unitree_g1_flat_toe_rectangle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """W1.6: 30 mm toe target during the planned mid-swing window."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  term = cfg.rewards["foot_clearance"]
  term.func = flat_toe_clearance
  term.params.update(
    target_height=0.03, period=0.6, offset=[0.0, 0.5],
    swing_phase_start=0.66, swing_phase_end=0.90,
  )
  return cfg


def unitree_g1_flat_toe_smooth_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """W1.8: same toe reward with a smooth single-peak time target."""
  cfg = unitree_g1_flat_toe_rectangle_env_cfg(play=play)
  cfg.rewards["foot_clearance"].params["target_shape"] = "smooth"
  return cfg
