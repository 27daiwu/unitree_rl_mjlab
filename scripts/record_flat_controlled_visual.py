"""Fixed-command Flat video using the A4 clean evaluator's control loop."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.mdp.navigation import evaluation_observations

from evaluate_g1_flat_control import COMMANDS


CHECKPOINTS = {
  'a4': 'logs/rsl_rl/g1_velocity/2026-09-28_20-30-16_phase_a4_flat_control/model_499.pt',
  'w1_3': 'logs/rsl_rl/g1_velocity/2026-09-30_12-22-22_flat_w1_3_continuation_control/model_399.pt',
}
SCENARIOS = {
  'forward': 'forward_020', 'backward': 'backward_015',
  'left': 'left_020', 'right': 'right_020',
  'turn_left': 'turn_left_050', 'turn_right': 'turn_right_050',
  'forward_fast': 'forward_040',
}


@dataclass
class Config:
  output_dir: str = 'logs/flat_w1_3_visual_gate'
  seed: int = 42
  num_envs: int = 4
  width: int = 960
  height: int = 720
  distance: float = 2.6
  elevation: float = -12.0
  azimuth: float = 45.0
  lookat_z: float = -0.8
  steps: int = 600
  scenario: str = 'all'
  checkpoint: str = 'all'
  checkpoint_control: str | None = None
  checkpoint_experiment: str | None = None


def run(cfg: Config):
  import src.tasks  # noqa: F401
  if bool(cfg.checkpoint_control) != bool(cfg.checkpoint_experiment):
    raise ValueError('Both matched checkpoint paths are required')
  checkpoints = ({'control': cfg.checkpoint_control, 'experiment': cfg.checkpoint_experiment}
                 if cfg.checkpoint_control else CHECKPOINTS)
  if cfg.scenario != 'all' and cfg.scenario not in SCENARIOS:
    raise ValueError(f'Unknown scenario: {cfg.scenario}')
  if cfg.checkpoint != 'all' and cfg.checkpoint not in checkpoints:
    raise ValueError(f'Unknown checkpoint: {cfg.checkpoint}')
  if not 1 <= cfg.steps <= 600:
    raise ValueError('steps must be 1..600')
  output = Path(cfg.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  records = []
  for label, checkpoint in checkpoints.items():
    if cfg.checkpoint not in ('all', label):
      continue
    env_cfg = load_env_cfg('Unitree-G1-Flat')
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = 1000.
    env_cfg.events['reset_base'].params['pose_range'] = {}
    env_cfg.events = {k: v for k, v in env_cfg.events.items()
                      if k in ('reset_base', 'reset_robot_joints')}
    env_cfg.curriculum = {}
    for group in env_cfg.observations.values():
      group.enable_corruption = False
    camera = env_cfg.viewer
    camera.width, camera.height = cfg.width, cfg.height
    camera.distance, camera.elevation, camera.azimuth = cfg.distance, cfg.elevation, cfg.azimuth
    camera.lookat = (0., 0., cfg.lookat_z)
    camera.max_extra_envs = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device='cuda:0', render_mode='rgb_array')
    try:
      rl = load_rl_cfg('Unitree-G1-Flat')
      vec = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
      runner = load_runner_cls('Unitree-G1-Flat')(vec, asdict(rl), device='cuda:0')
      runner.load(checkpoint, load_cfg={'actor': True}, strict=True, map_location='cuda:0')
      policy = runner.get_inference_policy(device='cuda:0')
      for short, scenario in SCENARIOS.items():
        if cfg.scenario not in ('all', short):
          continue
        command = COMMANDS[scenario]
        env.reset(seed=cfg.seed)
        term = env.command_manager.get_term('twist')
        term._update_command = lambda: None
        term.time_left.fill_(1e9)
        path = output / f'{label}_{short}.mp4'
        with imageio.get_writer(path, fps=50, codec='libx264', quality=8,
                                macro_block_size=16) as writer:
          for step in range(cfg.steps):
            term.vel_command_b[:] = term.vel_command_b.new_tensor(command)
            term.is_standing_env[:] = False
            term.is_heading_env[:] = False
            term.time_left.fill_(1e9)
            obs = evaluation_observations(env)
            with torch.no_grad():
              action = policy(obs)
              _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0] or truncated[0]):
              raise RuntimeError(f'{label} {scenario}: rendered replica terminated at step {step}')
            frame = env.render()
            if frame is None or not np.any(frame):
              raise RuntimeError('Blank render')
            writer.append_data(frame)
        record = dict(checkpoint_label=label, checkpoint=str(Path(checkpoint).resolve()),
                      scenario=scenario, command=command, seed=cfg.seed,
                      num_envs=cfg.num_envs, rendered_env=0, steps=cfg.steps,
                      duration_s=cfg.steps*env.step_dt, fps=50,
                      resolution=[cfg.width, cfg.height],
                      camera=dict(distance=cfg.distance, elevation=cfg.elevation,
                                  azimuth=cfg.azimuth, lookat=camera.lookat,
                                  origin_type=str(camera.origin_type), body_name=camera.body_name),
                      video=str(path))
        records.append(record)
        (output/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        print(f'Recorded {path}', flush=True)
    finally:
      env.close()


if __name__ == '__main__':
  run(tyro.cli(Config))
