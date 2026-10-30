import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_flat_gait import _capture, contact_labels, load_foot_collision_points


class CaptureTests(unittest.TestCase):
    def test_actuator_ids_do_not_replace_foot_site_ids(self):
        data = SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(1, 3), root_link_ang_vel_b=torch.zeros(1, 3),
            root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=torch.zeros(1, 4),
            site_pos_w=torch.arange(18).reshape(1, 6, 3),
            site_quat_w=torch.zeros(1, 6, 4), site_lin_vel_w=torch.zeros(1, 6, 3),
            joint_pos=torch.zeros(1, 29), joint_vel=torch.zeros(1, 29),
            joint_acc=torch.zeros(1, 29), actuator_force=torch.zeros(1, 29),
            soft_joint_pos_limits=torch.zeros(1, 29, 2),
        )
        sensor = SimpleNamespace(**{k: torch.zeros(1, 2) for k in (
            "current_contact_time", "current_air_time", "force", "force_history")})
        robot = SimpleNamespace(data=data, num_actuators=29,
            site_names=("a", "left_foot", "b", "right_foot", "c", "d"),
            actuators=[SimpleNamespace(ctrl_ids=torch.tensor([25, 26]),
                                      cfg=SimpleNamespace(effort_limit=10.0))])
        env = SimpleNamespace(scene={"robot": robot, "feet_ground_contact": SimpleNamespace(data=sensor)},
            num_envs=1, device="cpu", action_manager=SimpleNamespace(action=torch.zeros(1, 29)),
            reward_manager=SimpleNamespace(_step_reward=torch.zeros(1, 1), active_terms=("test",)))
        _capture(env)
        self.assertTrue(torch.equal(env.gait_diag["site"], data.site_pos_w[:, [1, 3]]))
        self.assertEqual(env.gait_diag["actuator_limit"][0, 25].item(), 10.0)


class ContactClassificationTests(unittest.TestCase):
    def test_force_onset_after_swing_is_touchdown_candidate(self):
        time = np.arange(8) * 0.02
        force = np.array([0, 0, 0, 0, 0, 0, 12, 18], dtype=float)
        events = contact_labels(np.zeros(8), force, time)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["classification"], "touchdown_candidate")
        self.assertAlmostEqual(events[0]["preceding_no_force_s"], 0.12)

    def test_short_force_contact_is_not_mislabeled_as_touchdown(self):
        time = np.arange(8) * 0.02
        force = np.array([0, 0, 0, 10, 0, 0, 0, 0], dtype=float)
        events = contact_labels(np.zeros(8), force, time)
        self.assertEqual(events[0]["classification"], "brief_contact")

    def test_low_height_without_force_does_not_create_contact_event(self):
        time = np.arange(8) * 0.02
        events = contact_labels(np.zeros(8), np.zeros(8), time)
        self.assertEqual(events, [])

    def test_force_contact_while_collision_envelope_is_clear_is_swing_candidate(self):
        time = np.arange(4) * 0.005
        force = np.array([0, 0, 14, 20], dtype=float)
        events = contact_labels(np.zeros(4), force, time, clearance=np.full(4, 0.025))
        self.assertEqual(events[0]["classification"], "swing_phase_contact_candidate")


class FootGeometryTests(unittest.TestCase):
    def test_collision_capsule_geometry_is_read_for_both_feet(self):
        path = Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_g1/xmls/g1.xml"
        points = load_foot_collision_points(path)
        self.assertEqual(set(points), {"left", "right"})
        for foot_points in points.values():
            self.assertTrue(np.any(foot_points[:, 0] > 0.05))
            self.assertTrue(np.any(foot_points[:, 0] < -0.03))
            self.assertAlmostEqual(float(np.max(foot_points[:, 2])), 0.02, places=6)


if __name__ == "__main__":
    unittest.main()
