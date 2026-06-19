"""Small checks for the methodological mistakes most likely during a refactor."""
import unittest
import numpy as np
from foodvision.features import meal_representations, normalize_embeddings
from foodvision.models import CandidateRegressors, group_weights, lower_weighted_median
from foodvision.benchmark import check_partition, group_mae

class MethodChecks(unittest.TestCase):

    def test_zero_embeddings_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_embeddings(np.zeros((1, 3)))

    def test_identical_photos_have_zero_change(self):
        photos = np.array([[1, 2, 3], [3, 1, 2]])
        features = meal_representations(photos, photos)
        np.testing.assert_allclose(features['Vector change'], 0, atol=1e-14)
        np.testing.assert_allclose(features['Cosine change'], 0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(features['Paired'], axis=1), 1)

    def test_groups_have_equal_total_weight(self):
        groups = np.array(['small', 'large', 'large', 'large'])
        weights = group_weights(groups)
        self.assertAlmostEqual(weights[0], weights[1:].sum())

    def test_median_tie_uses_lower_endpoint(self):
        self.assertEqual(lower_weighted_median(np.array([0.1, 0.9]), np.ones(2)), 0.1)

    def test_large_groups_do_not_dominate_scoring(self):
        predictions = np.array([[1, 0, 0, 0]])
        score = group_mae(predictions, np.zeros(4), np.array(['a', 'b', 'b', 'b']))
        self.assertAlmostEqual(score[0], 0.5)

    def test_similarity_links_block_different_groups(self):
        rows = [{'record_id': 'one', 'group': 'a', 'similarity_component': 'shared'}, {'record_id': 'two', 'group': 'b', 'similarity_component': 'shared'}]
        with self.assertRaises(ValueError):
            check_partition(rows, [0], [1])

    def test_empty_training_uses_zero_not_evaluation_labels(self):
        features = meal_representations(np.eye(2), np.eye(2))
        regressors = CandidateRegressors(features, [0.2, 0.8], ['a', 'b'])
        np.testing.assert_array_equal(regressors.predict([], [0, 1]).predictions, 0)
if __name__ == '__main__':
    unittest.main()
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from foodvision.benchmark import verify_payload
import unittest

class PayloadChecks(unittest.TestCase):

    def test_changed_and_missing_inputs_fail(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            asset = root / 'data/payload.bin'
            asset.write_bytes(b'input')
            (root / 'data/checksums.json').write_text(json.dumps({'data/payload.bin': hashlib.sha256(b'input').hexdigest()}))
            self.assertEqual(verify_payload(root), 1)
            asset.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_payload(root)
            asset.unlink()
            with self.assertRaises(ValueError):
                verify_payload(root)
