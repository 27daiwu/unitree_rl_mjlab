from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
    unitree_g1_walk_env_cfg,
    unitree_g1_walk_realsense_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
<<<<<<< HEAD:mjlab/tasks/velocity/config/g1/__init__.py
    task_id="Mjlab-Velocity-Rough-Unitree-G1",
    env_cfg=unitree_g1_rough_env_cfg(),
    play_env_cfg=unitree_g1_rough_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Unitree-G1",
    env_cfg=unitree_g1_flat_env_cfg(),
    play_env_cfg=unitree_g1_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Walk-Unitree-G1",
    env_cfg=unitree_g1_walk_env_cfg(),
    play_env_cfg=unitree_g1_walk_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Realsen-Walk-Unitree-G1",
    env_cfg=unitree_g1_walk_realsense_cfg(),
    play_env_cfg=unitree_g1_walk_realsense_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
=======
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
>>>>>>> upstream/main:src/tasks/velocity/config/g1/__init__.py
)
