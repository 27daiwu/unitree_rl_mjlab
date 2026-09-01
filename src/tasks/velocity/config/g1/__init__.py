from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from src.tasks.velocity.walk_env_cfg import unitree_g1_walk_env_cfg

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg
from .stairs_baseline_env_cfg import unitree_g1_stairs_baseline_env_cfg
from .stairs_clearance_ablation_env_cfg import (
  unitree_g1_stairs_clearance_ablation_env_cfg,
)
from .stairs_env_cfg import unitree_g1_stairs_env_cfg
from .phase2c0_flat_sanity_env_cfg import unitree_g1_phase2c0_flat_sanity_env_cfg
from .phase2c1_mild_env_cfg import unitree_g1_phase2c1_mild_env_cfg
from .phase2c2_single_riser_env_cfg import unitree_g1_phase2c2_single_riser_env_cfg

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Walk",
  env_cfg=unitree_g1_walk_env_cfg(),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs",
  env_cfg=unitree_g1_stairs_env_cfg(),
  play_env_cfg=unitree_g1_stairs_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs-Baseline",
  env_cfg=unitree_g1_stairs_baseline_env_cfg(),
  play_env_cfg=unitree_g1_stairs_baseline_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Stairs-Clearance-Ablation",
  env_cfg=unitree_g1_stairs_clearance_ablation_env_cfg(),
  play_env_cfg=unitree_g1_stairs_clearance_ablation_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Phase2C0-Flat-Sanity",
  env_cfg=unitree_g1_phase2c0_flat_sanity_env_cfg(),
  play_env_cfg=unitree_g1_phase2c0_flat_sanity_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Phase2C1-Mild-Terrain",
  env_cfg=unitree_g1_phase2c1_mild_env_cfg(),
  play_env_cfg=unitree_g1_phase2c1_mild_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Phase2C2-Single-Riser",
  env_cfg=unitree_g1_phase2c2_single_riser_env_cfg(),
  play_env_cfg=unitree_g1_phase2c2_single_riser_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
