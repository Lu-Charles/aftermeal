"""Regression checks for model packaging, image input and the local HTTP API."""
import base64
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from foodvision.benchmark import verify_payload
from foodvision.inference import Predictor, ROOT
from foodvision.server import make_handler

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

class HttpChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        handler = make_handler()
        handler.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        status, data = (response.status, response.read())
        connection.close()
        return (status, data)

    def test_example_prediction_endpoint(self):
        status, data = self.request('POST', '/api/predict', json.dumps({'example_id': 'L133'}))
        self.assertEqual(status, 200)
        self.assertAlmostEqual(json.loads(data)['estimated_fraction'], 0.04217373482549647, places=10)

    def test_outside_origin_is_rejected(self):
        status, _ = self.request('POST', '/api/predict', '{}', {'Origin': 'https://example.com'})
        self.assertEqual(status, 403)

    def test_path_traversal_and_private_payloads_are_not_served(self):
        for path in ('/web/%2e%2e/data/records.json', '/data/records.json', '/models/demo_model.npz'):
            status, _ = self.request('GET', path)
            self.assertEqual(status, 404)

    def test_invalid_json_and_image_encoding_return_client_errors(self):
        for body in ('not-json', '[]', json.dumps({'before': '###', 'after': '###'})):
            status, _ = self.request('POST', '/api/predict', body)
            self.assertEqual(status, 400)

    def test_missing_pair_is_rejected(self):
        status, _ = self.request('POST', '/api/predict', json.dumps({'before': base64.b64encode(b'abc').decode()}))
        self.assertEqual(status, 400)
if __name__ == '__main__':
    unittest.main()
