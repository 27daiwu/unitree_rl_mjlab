from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from src.tasks.velocity.walk_env_cfg import unitree_g1_walk_env_cfg

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .flat_toe_env_cfgs import (
  unitree_g1_flat_toe_rectangle_env_cfg,
  unitree_g1_flat_toe_smooth_env_cfg,
)
from .flat_right_track_env_cfg import unitree_g1_flat_right_track_env_cfg
from .rl_cfg import unitree_g1_ppo_runner_cfg
from .stairs_baseline_env_cfg import unitree_g1_stairs_baseline_env_cfg
from .stairs_clearance_ablation_env_cfg import (
  unitree_g1_stairs_clearance_ablation_env_cfg,
)
from .stairs_env_cfg import unitree_g1_stairs_env_cfg
from .visual_stairs_v1a_env_cfg import (
  unitree_g1_visual_stairs_v1a_env_cfg,
  unitree_g1_visual_stairs_v1a_ppo_runner_cfg,
)

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
  task_id="Unitree-G1-Flat-W1.6-Toe-Rectangle",
  env_cfg=unitree_g1_flat_toe_rectangle_env_cfg(),
  play_env_cfg=unitree_g1_flat_toe_rectangle_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-W1.8-Toe-Smooth",
  env_cfg=unitree_g1_flat_toe_smooth_env_cfg(),
  play_env_cfg=unitree_g1_flat_toe_smooth_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat-W1.11-Right-Track",
  env_cfg=unitree_g1_flat_right_track_env_cfg(),
  play_env_cfg=unitree_g1_flat_right_track_env_cfg(play=True),
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
  task_id="Unitree-G1-Visual-Stairs-V1A",
  env_cfg=unitree_g1_visual_stairs_v1a_env_cfg(),
  play_env_cfg=unitree_g1_visual_stairs_v1a_env_cfg(play=True),
  rl_cfg=unitree_g1_visual_stairs_v1a_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Independent flat navigation baseline; existing tasks retain their configs.
from .navigation_env_cfg import (
  unitree_g1_navigation_env_cfg,
  unitree_g1_navigation_ppo_runner_cfg,
)
register_mjlab_task(
  task_id="Unitree-G1-Navigation",
  env_cfg=unitree_g1_navigation_env_cfg(),
  play_env_cfg=unitree_g1_navigation_env_cfg(play=True),
  rl_cfg=unitree_g1_navigation_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
