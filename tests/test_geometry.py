import unittest
import numpy as np
from steer.geometry.measure import distance, angle, dihedral, angular_difference
from steer.geometry.align import kabsch, anchored_rmsd


class GeometryTests(unittest.TestCase):
    def test_distance_known(self):
        self.assertEqual(distance([0, 0, 0], [3, 4, 0]), 5)

    def test_angle_known(self):
        self.assertAlmostEqual(angle([1, 0, 0], [0, 0, 0], [0, 1, 0]), 90)

    def test_dihedral_signed(self):
        self.assertAlmostEqual(dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]), -90)

    def test_periodic_boundary(self):
        self.assertEqual(angular_difference(179, -179), -2)
        self.assertEqual(angular_difference(-179, 179), 2)

    def test_degenerate_angle(self):
        with self.assertRaises(ValueError):
            angle([0, 0, 0], [0, 0, 0], [0, 1, 0])

    def test_degenerate_dihedral(self):
        with self.assertRaises(ValueError):
            dihedral([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0])

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            distance([float('nan'), 0, 0], [0, 0, 0])

    def test_rigid_invariance(self):
        x = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        y = x @ rotation + [10, -5, 1]
        aligned, rmsd, fitted, _ = kabsch(x, y)
        self.assertLess(rmsd, 1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted), 1)
        np.testing.assert_allclose(aligned, y, atol=1e-12)

    def test_no_reflection_alignment(self):
        x = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        y = x * [-1, 1, 1]
        self.assertGreater(kabsch(x, y)[1], .1)

    def test_anchored_fit_preserves_pose_error(self):
        anchors = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0]])
        self.assertAlmostEqual(anchored_rmsd(anchors, anchors, [[1, 1, 2]], [[1, 1, 1]]), 1)

    def test_collinear_anchors_rejected(self):
        x = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        with self.assertRaises(ValueError):
            anchored_rmsd(x, x, [[1, 1, 2]], [[1, 1, 1]])
