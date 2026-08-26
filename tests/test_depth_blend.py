import unittest
import numpy as np
from foodvision.depth_blend import choose_weight, combine, forecast
from foodvision.features import meal_representations

class BlendingChecks(unittest.TestCase):

    def test_missing_geometry_and_zero_weight_preserve_baseline(self):
        base, depth = (np.array([0.2, 0.7]), np.array([0.9, 0.1]))
        for policy in ('Blend', 'BoundedDownward'):
            np.testing.assert_array_equal(combine(base, depth, [False, False], 1, policy), base)
            np.testing.assert_array_equal(combine(base, depth, [True, True], 0, policy), base)

    def test_downward_policy_never_increases_or_exceeds_correction_cap(self):
        base, depth = (np.array([0.01, 0.8, 0.6]), np.array([0.4, 0.0, 0.55]))
        actual = combine(base, depth, [True, True, True], 0.5, 'BoundedDownward')
        np.testing.assert_allclose(actual, [0.01, 0.75, 0.575])
        self.assertTrue(np.all(actual <= base))
        self.assertTrue(np.all(base - actual <= 0.05 + 1e-12))

    def test_equal_experts_choose_zero_weight(self):
        base = np.array([0.0, 0.5])
        weight, details = choose_weight(base, base, [True, True], base, np.array(['a', 'b']), 'Blend')
        self.assertEqual(weight, 0)
        self.assertEqual(details['inner_small_positive_count'], 0)

    def test_zero_mass_guard_can_reject_improved_average(self):
        base, depth, target = (np.array([0.0, 1.0]), np.array([0.3, 0.5]), np.array([0.0, 0.5]))
        weight, details = choose_weight(base, depth, [True, True], target, np.array(['a', 'b']), 'Blend')
        self.assertFalse(details['admissible'][-1])
        self.assertLessEqual(weight, 0.25)

    def test_outer_labels_cannot_change_experts_weights_or_predictions(self):
        rng = np.random.default_rng(11)
        reps = meal_representations(rng.normal(size=(8, 5)), rng.normal(size=(8, 5)))
        features = {'a': rng.normal(size=(8, 4)), 'b': rng.normal(size=(8, 7))}
        targets = rng.uniform(size=8)
        groups = np.array([str(i) for i in range(8)])
        records = [{'record_id': str(i), 'group': str(i), 'similarity_component': str(i)} for i in range(8)]
        e = {'train': list(range(6)), 'test': [6, 7], 'budget': 6, 'inner': [{'train': [j for j in range(6) if j != i], 'validation': [i]} for i in range(6)]}
        a, selection_a = forecast(records, reps, features, np.ones(8, dtype=bool), targets, groups, e)
        targets[6:] = [-100, 100]
        b, selection_b = forecast(records, reps, features, np.ones(8, dtype=bool), targets, groups, e)
        self.assertEqual(selection_a, selection_b)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
