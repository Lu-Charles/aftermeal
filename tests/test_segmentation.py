"""Leakage, image identity, and degenerate-mask regression checks."""
import unittest
import numpy as np
from foodvision.segmentation import mask_features, pair_features
from foodvision.segmentation_experiment import check_boundaries, nested_forecast, standardized
from scripts.build_review_queue import assemble

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

    def test_evaluation_features_cannot_change_training_scaling(self):
        x = np.array([[0, 2], [2, 2], [3, 2.0]])
        a, _ = standardized(x, np.zeros((2, 2)))
        b, _ = standardized(x, np.full((2, 2), 1000000000.0))
        np.testing.assert_array_equal(a, b)

    def test_outer_labels_cannot_change_segmentation_selection(self):
        x = np.random.default_rng(8).normal(size=(7, 19))
        y = np.array([0, 0.2, 0.4, 0.6, 0.8, 0, 1.0])
        groups = np.array(list('abcdefg'))
        episode = {'train': list(range(5)), 'test': [5, 6], 'inner': [{'train': [j for j in range(5) if j != i], 'validation': [i]} for i in range(5)]}
        a, selection_a = nested_forecast(x, y, groups, episode)
        y[5:] = [1, 0]
        b, selection_b = nested_forecast(x, y, groups, episode)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(selection_a, selection_b)

    def test_shared_image_across_before_after_roles_is_rejected(self):
        records = [{'record_id': str(i), 'group': str(i), 'similarity_component': str(i), 'dependency_block': 'broad'} for i in range(2)]
        provenance = [{'before': {'sha256': 'a'}, 'after': {'sha256': 'b'}}, {'before': {'sha256': 'c'}, 'after': {'sha256': 'a'}}]
        with self.assertRaisesRegex(ValueError, 'original image'):
            check_boundaries(records, provenance, [0], [1])
        provenance[1]['after']['sha256'] = 'd'
        self.assertEqual(check_boundaries(records, provenance, [0], [1]), ['broad'])

    def test_ai_proposals_cannot_arrive_as_gold(self):
        with self.assertRaisesRegex(ValueError, 'unverified AI'):
            assemble({'human_review': 'pending', 'evaluation_gold': True, 'training_eligible': False}, [], {})

    def test_incomplete_review_sheet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'every cell'):
            assemble({'human_review': 'pending', 'evaluation_gold': False, 'training_eligible': False, 'pages': {'1': 'F'}}, [{}, {}], {})
if __name__ == '__main__':
    unittest.main()
