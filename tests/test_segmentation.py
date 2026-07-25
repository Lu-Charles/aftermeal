"""Leakage, image identity, and degenerate-mask regression checks."""
import unittest
import numpy as np
from foodvision.segmentation import mask_features, pair_features

class SegmentationChecks(unittest.TestCase):

    def test_empty_starting_mask_does_not_produce_infinity(self):
        empty = mask_features(np.zeros((4, 4)), np.zeros((4, 4)))
        full = mask_features(np.ones((4, 4)), np.ones((4, 4)))
        features = pair_features(np.array([empty, empty]), np.array([empty, full]))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:, -1], [0, 2])
        np.testing.assert_array_equal(empty[[0, 3, 4, 5]], [0, 0, 0.5, 0.5])

    def test_invalid_probability_is_rejected(self):
        for probability in (np.full((2, 2), np.nan), np.full((2, 2), 1.1)):
            with self.assertRaises(ValueError):
                mask_features(np.zeros((2, 2)), probability)
if __name__ == '__main__':
    unittest.main()
