"""Small checks for the methodological mistakes most likely during a refactor."""
import unittest
from foodvision.benchmark import check_partition

class MethodChecks(unittest.TestCase):

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
