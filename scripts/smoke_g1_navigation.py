"""CUDA integration check with an untrained policy; never calls learn()."""
from dataclasses import asdict
import json

import torch

from src.tasks.velocity.mdp.navigation import evaluation_observations
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import src.tasks  # noqa: F401


def main():
  task = "Unitree-G1-Navigation"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 4
  cfg.seed = 42
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl_cfg = load_rl_cfg(task)
    vec = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = load_runner_cls(task)(vec, asdict(rl_cfg), device="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    term = env.command_manager.get_term("twist")
    sequence = [(0.05, 0., 0.), (0., 0., 0.), (.2, 0., 0.), (-.1, 0., 0.),
                (0., .1, 0.), (0., -.1, 0.), (0., 0., .4), (0., 0., 0.),
                (.15, .1, .3), (.05, -.1, -.3)]
    for command in sequence:
      term.set_evaluation_command(command)
      for _ in range(5):
        obs = evaluation_observations(env)
        # Verify the policy sees the NEW command immediately, not cached history.
        names = env.observation_manager.active_terms["actor"]
        dims = env.observation_manager.group_obs_term_dim["actor"]
        start = sum(dim[0] for dim in dims[:names.index("command")])
        torch.testing.assert_close(obs["actor"][:, start:start+3], term.command)
        with torch.no_grad():
          action = policy(obs)
          _, reward, _, _, _ = env.step(action)
        assert torch.isfinite(action).all() and torch.isfinite(reward).all()
        torch.testing.assert_close(term.command, torch.tensor(command, device=env.device).expand(4, -1))
    # Return to training sampling and cross resampling boundaries, including reset.
    term.evaluation_command = None
    env.reset()
    for _ in range(170):
      obs = evaluation_observations(env)
      with torch.no_grad():
        _, reward, _, _, _ = env.step(policy(obs))
      assert torch.isfinite(reward).all()
    print(json.dumps(dict(device=torch.cuda.get_device_name(0), envs=4,
                          steps=220, transitions=9, finite=True, training_started=False)))
  finally:
    env.close()


if __name__ == "__main__":
  main()
