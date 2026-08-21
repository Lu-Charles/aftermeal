"""Check geometric invariants and fitting boundaries, without downloading weights."""
import unittest
import numpy as np
from foodvision.depth import pair_features, shape_features
from foodvision.segmentation_experiment import nested_forecast

class DepthFeatures(unittest.TestCase):

    def setUp(self):
        self.y, self.x = np.mgrid[-1:1:64j, -1:1:64j]
        self.mask = self.x ** 2 + self.y ** 2 < 0.25
        self.plane = 3 + 0.2 * self.x + 0.1 * self.y

    def test_plane_has_no_artificial_food_height(self):
        features = shape_features(self.plane, self.mask)
        self.assertEqual(features[1], 1)
        np.testing.assert_allclose(features[2:8], 0, atol=1e-12)

    def test_raised_region_differs_from_flat_residue(self):
        flat = shape_features(self.plane, self.mask)
        raised = shape_features(self.plane + self.mask * 0.5, self.mask)
        self.assertGreater(raised[2], flat[2])
        self.assertGreater(raised[5], flat[5])

    def test_positive_depth_scale_and_shift_do_not_change_features(self):
        depth = self.plane + self.mask * 0.5
        np.testing.assert_allclose(shape_features(depth, self.mask), shape_features(depth * 7 + 13, self.mask), atol=1e-12)

    def test_empty_or_full_mask_explicitly_has_no_reference(self):
        for mask in (np.zeros_like(self.mask), np.ones_like(self.mask)):
            features = shape_features(self.plane, mask)
            self.assertTrue(np.isfinite(features).all())
            self.assertEqual(features[1], 0)
            np.testing.assert_array_equal(features[2:8], 0)

    def test_constant_depth_has_finite_zero_residuals(self):
        features = shape_features(np.ones_like(self.plane), self.mask)
        self.assertEqual(features[1], 1)
        np.testing.assert_array_equal(features[2:], 0)

    def test_invalid_depth_and_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            shape_features(self.plane, self.mask[:2])
        with self.assertRaises(ValueError):
            shape_features(self.plane * np.nan, self.mask)
        with self.assertRaises(ValueError):
            pair_features(np.zeros((1, 9)), np.zeros((2, 9)))

    def test_pair_features_retain_before_after_and_change(self):
        before, after = (np.ones((2, 9)), np.full((2, 9), 2.0))
        actual = pair_features(before, after)
        np.testing.assert_array_equal(actual, np.column_stack([before, after, after - before]))

class DepthSelection(unittest.TestCase):

    def test_outer_labels_cannot_change_selection_or_prediction(self):
        rng = np.random.default_rng(6)
        x, y = (rng.normal(size=(8, 27)), rng.uniform(size=8))
        groups = np.array([str(i) for i in range(8)])
        e = {'train': list(range(6)), 'test': [6, 7], 'inner': [{'train': [j for j in range(6) if j != i], 'validation': [i]} for i in range(6)]}
        prediction, selection = nested_forecast(x, y, groups, e)
        y[6:] = [100, -100]
        changed, changed_selection = nested_forecast(x, y, groups, e)
        np.testing.assert_array_equal(prediction, changed)
        self.assertEqual(selection, changed_selection)
