"""Navigation-only commands and shaping; no terrain or disturbance coupling."""
from dataclasses import dataclass

import torch

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


def sample_navigation_commands(n, device, low_speed_probability=0.75):
  """Five equally likely modes; 75% of moving samples use the low-speed box."""
  low = torch.rand(n, device=device) < low_speed_probability
  mode = torch.randint(0, 5, (n,), device=device)
  u = torch.rand(n, 3, device=device)
  limits = torch.tensor([0.4, 0.25, 0.8], device=device).expand(n, -1).clone()
  limits[low] = torch.tensor([0.19, 0.14, 0.39], device=device)
  cmd = (2 * u - 1) * limits
  # Positive x is the core workspace; 10% of fresh x samples are backwards.
  cmd[:, 0] = u[:, 0] * limits[:, 0]
  reverse = torch.rand(n, device=device) < 0.1
  cmd[reverse, 0] *= -0.5
  cmd[mode == 0] = 0
  cmd[mode == 1, 1:] = 0
  cmd[mode == 2, 0] = 0
  cmd[mode == 2, 2] = 0
  cmd[mode == 3, :2] = 0
  return cmd


class NavigationVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.evaluation_command = None

  def set_evaluation_command(self, command):
    """Pin command before observation computation and across env.step()."""
    self.evaluation_command = torch.as_tensor(command, device=self.device).expand_as(self.vel_command_b).clone()
    self.vel_command_b.copy_(self.evaluation_command)

  def _resample_command(self, env_ids):
    previous = self.vel_command_b[env_ids].clone()
    command = sample_navigation_commands(len(env_ids), self.device, self.cfg.low_speed_probability)
    # _resample increments counter AFTER calling this method; reset starts at 0.
    paired = self.command_counter[env_ids] % 2 == 1
    reverse = paired & (torch.rand(len(env_ids), device=self.device) < 0.5)
    command[paired] = 0  # move/side-step/turn -> stop
    command[reverse] = -previous[reverse]
    command[:, 0].clamp_(-0.2, 0.4)
    self.vel_command_b[env_ids] = command
    self.is_standing_env[env_ids] = (command == 0).all(dim=1)
    self.is_heading_env[env_ids] = False

  def _update_command(self):
    if self.evaluation_command is not None:
      self.vel_command_b.copy_(self.evaluation_command)


@dataclass(kw_only=True)
class NavigationVelocityCommandCfg(UniformVelocityCommandCfg):
  low_speed_probability: float = 0.75

  def build(self, env):
    return NavigationVelocityCommand(self, env)


def navigation_phase(env, period, command_name):
  phase = env.episode_length_buf * env.step_dt * (2 * torch.pi / period)
  result = torch.stack((phase.sin(), phase.cos()), dim=1)
  moving = env.command_manager.get_command(command_name).abs().amax(dim=1) > 1e-6
  return result * moving[:, None]


def axis_tracking(env, axis, std, command_name="twist"):
  data = env.scene["robot"].data
  actual = data.root_link_lin_vel_b[:, axis] if axis < 2 else data.root_link_ang_vel_b[:, 2]
  error = actual - env.command_manager.get_command(command_name)[:, axis]
  return torch.exp(-error.square() / std**2)


def velocity_error(env, command_name="twist", zero_only=False, turn_only=False):
  data = env.scene["robot"].data
  command = env.command_manager.get_command(command_name)
  actual = torch.cat((data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2:3]), dim=1)
  error = ((actual - command) / actual.new_tensor([0.2, 0.15, 0.4])).square()
  if turn_only:
    return error[:, :2].sum(1) * ((command[:, :2].abs().amax(1) < 1e-6) & (command[:, 2].abs() > 1e-6))
  cost = error.sum(1)
  if zero_only:
    cost *= command.abs().amax(1) < 1e-6
  return cost


def evaluation_observations(env):
  """Refresh command/phase in cached observations without advancing sensor delays.

  mjlab compute(update_history=False) returns a cache. Recomputing all terms
  instead would double-push Phase B delay buffers. Navigation control signals
  intentionally have no noise, delay or stacked history.
  """
  import math
  manager = env.observation_manager
  result = {key: value.clone() for key, value in manager.compute().items()}
  for group, names in manager.active_terms.items():
    offset = 0
    for name, dim in zip(names, manager.group_obs_term_dim[group], strict=True):
      size = math.prod(dim)
      if name in ("command", "phase"):
        cfg = manager.get_term_cfg(group, name)
        value = cfg.func(env, **cfg.params)
        if cfg.delay_max_lag or size != value.shape[1] or cfg.scale is not None or cfg.clip is not None:
          raise ValueError("Navigation command/phase must remain immediate, unscaled, single-frame control signals")
        result[group][:, offset:offset+size] = value
      offset += size
  return result


def vertical_velocity(env):
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()
