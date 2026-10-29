import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_flat_gait import contact_labels, load_foot_collision_points


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
