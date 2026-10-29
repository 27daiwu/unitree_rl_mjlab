"""Run with python -m unittest discover -s tests -p 'test_navigation.py'."""
import unittest
import math
from types import SimpleNamespace

import numpy as np
import torch

from src.tasks.velocity.config.g1.navigation_env_cfg import unitree_g1_navigation_env_cfg
from src.tasks.velocity.mdp.navigation import NavigationVelocityCommand, sample_navigation_commands, axis_tracking, velocity_error
from src.tasks.velocity.mdp.navigation_eval import DisturbanceWindow, recovery_time, summarize_trace


class NavigationTests(unittest.TestCase):
  def test_sampling_workspace_and_low_speed(self):
    torch.manual_seed(42)
    c = sample_navigation_commands(50000, "cpu")
    self.assertTrue(((c >= torch.tensor([-.2, -.25, -.8])) & (c <= torch.tensor([.4, .25, .8]))).all())
    moving = c.abs().amax(1) > 0
    low = (c.abs() < torch.tensor([.2, .15, .4])).all(1)
    self.assertGreater(low[moving].float().mean(), .7)
    self.assertGreater(((c[:, 0] > 0) & (c[:, 0] < .05)).sum(), 100)
    self.assertTrue(((c[:, :2] == 0).all(1) & (c[:, 2] != 0)).any())
    self.assertTrue((c != 0).all(1).any())

  def test_paired_transitions_and_reset(self):
    # Exercise actual resampling without simulation or root-state writes.
    term = NavigationVelocityCommand.__new__(NavigationVelocityCommand)
    term._env = SimpleNamespace(num_envs=5000, device="cpu")
    term.cfg = SimpleNamespace(low_speed_probability=.75)
    term.vel_command_b = torch.zeros(5000, 3)
    term.is_standing_env = torch.zeros(5000, dtype=torch.bool)
    term.is_heading_env = term.is_standing_env.clone()
    term.command_counter = torch.zeros(5000, dtype=torch.long)
    ids = torch.arange(5000)
    term._resample_command(ids)
    old = term.command.clone()
    term.command_counter += 1
    term._resample_command(ids)
    new = term.command
    self.assertGreater(((old.abs().sum(1) > 0) & (new == 0).all(1)).sum(), 100)
    for axis in range(3):
      self.assertGreater(((old[:, axis] > 0) & (new[:, axis] < 0)).sum(), 100)
    term.command_counter.zero_()
    term._resample_command(ids)
    self.assertGreater((term.command[:, 0] > 0).sum(), 100)

  def test_config_isolation(self):
    from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    cfg = unitree_g1_navigation_env_cfg()
    self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
    self.assertFalse(cfg.curriculum)
    self.assertEqual(set(cfg.events), {"reset_base", "reset_robot_joints"})
    self.assertNotIn("height_scan", cfg.observations["actor"].terms)
    self.assertIn("push_robot", unitree_g1_flat_env_cfg().events)
    self.assertEqual(cfg.rewards["stand_still"].params["command_threshold"], 1e-6)

  def test_reward_precision_and_stop(self):
    cmd = torch.tensor([[.05, 0., 0.], [0., 0., 0.], [0., 0., .4]])
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[.05, 0., 0.], [.1, 0., 0.], [.1, 0., 0.]]), root_link_ang_vel_b=torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., .4]]))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, command_manager=SimpleNamespace(get_command=lambda _: cmd))
    self.assertEqual(axis_tracking(env, 0, .1)[0], 1.)
    self.assertGreater(velocity_error(env, zero_only=True)[1], 0)
    self.assertEqual(velocity_error(env, zero_only=True)[0], 0)
    self.assertGreater(velocity_error(env, turn_only=True)[2], 0)

  def test_axis_tracking_reward_kernel_values_and_bounds(self):
    for axis, command, sigma in ((0, .2, .1), (0, .4, .35),
                                 (1, .2, .08), (2, .5, .2)):
      c = torch.zeros(1, 3)
      c[0, axis] = command
      data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(1, 3),
        root_link_ang_vel_b=torch.zeros(1, 3),
      )
      env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: c),
      )
      actual = axis_tracking(env, axis, sigma).item()
      self.assertAlmostEqual(actual, math.exp(-(command / sigma) ** 2), places=6)
      self.assertTrue(0 <= actual <= 1)
      if axis < 2:
        data.root_link_lin_vel_b[0, axis] = command
      else:
        data.root_link_ang_vel_b[0, axis] = command
      self.assertEqual(axis_tracking(env, axis, sigma).item(), 1.)

  def test_recovery_requires_contiguous_hold(self):
    t = np.arange(1, 101) * .02
    e = np.zeros((100, 3)); r = e[:, :2].copy()
    e[:20, 0] = .2
    r[30, 0] = .3  # briefly outside attitude tolerance resets hold
    self.assertAlmostEqual(recovery_time(t, e, r, .2), .44)
    e[-70:, 0] = .2
    self.assertIsNone(recovery_time(t, e, r, .2))

  def test_metrics_known_motion_and_disturbance(self):
    t = np.arange(1, 201) * .02
    c = np.zeros((200, 3)); c[:100, 0] = .1
    v = c.copy(); p = np.zeros_like(c); p[:, 0] = np.cumsum(v[:, 0]) * .02
    trace = dict(time=t, command=c, velocity=v, position=p, rpy=np.zeros_like(c), initial_yaw=0., initial_position=np.zeros(3))
    result = summarize_trace(trace, disturbances=(DisturbanceWindow("push", .2, .4),))
    np.testing.assert_allclose(result["velocity_rmse"], 0)
    self.assertAlmostEqual(result["segments"][1]["stopping_distance"], 0)
    self.assertAlmostEqual(result["segments"][1]["post_stop_drift"], 0)
    self.assertAlmostEqual(result["robustness"][0]["post_disturbance_lateral_drift"], 0)
    self.assertIsNotNone(result["robustness"][0]["recovery_time"])
    v[:, 1] = .1
    result = summarize_trace(trace, fallen=True)
    self.assertAlmostEqual(result["velocity_rmse"][1], .1)
    self.assertEqual(result["fall_rate"], 1.)
    self.assertIsNone(result["segments"][1]["recovery_time"])


  def test_unwrapped_turning_error(self):
    t = np.arange(1, 1001) * .02
    c = np.zeros((1000, 3)); c[:, 2] = .4
    rpy = np.zeros_like(c); rpy[:, 2] = (.4*t + np.pi) % (2*np.pi) - np.pi
    trace = dict(time=t, command=c, velocity=c.copy(), position=np.zeros_like(c), rpy=rpy, initial_yaw=0., initial_position=np.zeros(3))
    result = summarize_trace(trace)
    self.assertAlmostEqual(result["segments"][0]["turning_error"], 0)
    self.assertAlmostEqual(result["heading_deviation"], 0)

  def test_robustness_drift_subtracts_commanded_lateral_motion(self):
    t = np.arange(1, 201) * .02
    c = np.zeros((200, 3)); c[:, 1] = .1
    p = np.zeros_like(c); p[:, 1] = .1*t
    trace = dict(time=t, command=c, velocity=c.copy(), position=p, rpy=np.zeros_like(c), initial_yaw=0., initial_position=np.zeros(3))
    result = summarize_trace(trace, disturbances=(DisturbanceWindow("friction", .2, .4),))
    self.assertAlmostEqual(result["robustness"][0]["post_disturbance_lateral_drift"], 0)
    p[:, 1] += .05*t
    result = summarize_trace(trace, disturbances=(DisturbanceWindow("friction", .2, .4),))
    self.assertAlmostEqual(result["robustness"][0]["post_disturbance_lateral_drift"], .18)


if __name__ == "__main__":
  unittest.main()
