"""Small checks for the methodological mistakes most likely during a refactor."""
import unittest
import numpy as np
from foodvision.models import group_weights, lower_weighted_median
from foodvision.benchmark import check_partition

class MethodChecks(unittest.TestCase):

    def test_groups_have_equal_total_weight(self):
        groups = np.array(['small', 'large', 'large', 'large'])
        weights = group_weights(groups)
        self.assertAlmostEqual(weights[0], weights[1:].sum())

    def test_median_tie_uses_lower_endpoint(self):
        self.assertEqual(lower_weighted_median(np.array([0.1, 0.9]), np.ones(2)), 0.1)

    def test_similarity_links_block_different_groups(self):
        rows = [{'record_id': 'one', 'group': 'a', 'similarity_component': 'shared'}, {'record_id': 'two', 'group': 'b', 'similarity_component': 'shared'}]
        with self.assertRaises(ValueError):
            check_partition(rows, [0], [1])
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
