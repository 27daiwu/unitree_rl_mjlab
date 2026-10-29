"""Serialize resolved runtime configuration without mutating task behavior."""
from dataclasses import fields, is_dataclass, asdict
from pathlib import Path
import importlib.metadata
import json
import torch

ROOT = Path('logs/navigation_phase_a4')


def encode(x):
  if is_dataclass(x): return {f.name: encode(getattr(x, f.name)) for f in fields(x)}
  if isinstance(x, dict): return {str(k): encode(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)): return [encode(v) for v in x]
  if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
  if callable(x): return x.__module__ + '.' + x.__qualname__
  if x is None or isinstance(x, (str, int, float, bool)): return x
  return str(x)


def snapshot(env, runner, label):
  robot = env.scene['robot']; manager = env.observation_manager
  result = dict(config=encode(env.cfg), command_runtime=encode(env.command_manager.get_term('twist').cfg),
    observation_dimensions=encode(manager.group_obs_dim),
    observation_order=encode(manager.active_terms), observation_term_dimensions=encode(manager.group_obs_term_dim),
    action_dimension=env.action_manager.total_action_dim,
    joint_order=list(robot.joint_names), actuator_order=list(robot.actuator_names),
    actor=str(runner.alg.actor), critic=str(runner.alg.critic),
    actor_state_shapes={k:list(v.shape) for k,v in runner.alg.actor.state_dict().items()},
    default_root_state=encode(robot.data.default_root_state[0]),
    default_joint_position=encode(robot.data.default_joint_pos[0]),
    default_joint_velocity=encode(robot.data.default_joint_vel[0]),
    step_dt=env.step_dt, physics_dt=env.physics_dt,
    versions={k:importlib.metadata.version(k) for k in ['mjlab','mujoco','mujoco-warp','torch','rsl-rl-lib']})
  ROOT.mkdir(parents=True,exist_ok=True)
  (ROOT/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
  return result


def main():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
  import src.tasks
  for task,label in [('Unitree-G1-Flat','flat_preflight')]:
    cfg=load_env_cfg(task);cfg.scene.num_envs=4;cfg.seed=42
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
      rl=load_rl_cfg(task);runner=load_runner_cls(task)(RslRlVecEnvWrapper(env,clip_actions=rl.clip_actions),asdict(rl),device='cuda:0')
      env.reset(seed=42)
      snapshot(env,runner,label)
      (ROOT/(label+'_ppo.json')).write_text(json.dumps(encode(rl),indent=2)+'\n')
    finally:env.close()


if __name__=='__main__':main()
