"""Regression checks for model packaging, image input and the local HTTP API."""
import json
import unittest
from foodvision.benchmark import verify_payload
from foodvision.inference import Predictor, ROOT

class DemoModelChecks(unittest.TestCase):

    def test_bundled_artifacts_match_their_checksums(self):
        self.assertGreater(verify_payload(), 10)

    def test_demo_examples_are_outside_model_calibration_groups(self):
        records = json.loads((ROOT / 'data/records.json').read_text())
        metadata = json.loads((ROOT / 'models/demo_model.json').read_text())
        examples = json.loads((ROOT / 'examples/examples.json').read_text())
        train = [r for r in records if r['record_id'] in metadata['train_ids']]
        test = [r for r in records if r['record_id'] in {e['id'] for e in examples}]
        self.assertEqual(len(train), 50)
        self.assertEqual(len(test), 4)
        for field in ('record_id', 'group', 'similarity_component'):
            self.assertFalse({r[field] for r in train} & {r[field] for r in test})

    def test_export_matches_all_four_saved_forecasts(self):
        predictor = Predictor()
        for example in json.loads((ROOT / 'examples/examples.json').read_text()):
            result = predictor.example(example['id'])
            self.assertAlmostEqual(result['estimated_fraction'], example['expected_prediction'], places=10)

    def test_unknown_example_is_rejected(self):
        with self.assertRaises(ValueError):
            Predictor().example('not-a-record')
if __name__ == '__main__':
    unittest.main()
