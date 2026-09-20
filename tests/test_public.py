"""Public deployment exposes samples only and enforces its smaller request contract."""
import io
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from wsgiref.validate import validator
from foodvision.public import PublicApplication

class PublicTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = PublicApplication()

    def request(self, method='GET', path='/', body=b'', **updates):
        environ = {'REQUEST_METHOD': method, 'PATH_INFO': path, 'SCRIPT_NAME': '', 'QUERY_STRING': '', 'SERVER_NAME': 'demo.example', 'SERVER_PORT': '443', 'SERVER_PROTOCOL': 'HTTP/1.1', 'HTTP_HOST': 'demo.example', 'CONTENT_TYPE': 'application/json', 'CONTENT_LENGTH': str(len(body)), 'wsgi.input': io.BytesIO(body), 'wsgi.errors': io.StringIO(), 'wsgi.version': (1, 0), 'wsgi.url_scheme': 'https', 'wsgi.multithread': False, 'wsgi.multiprocess': True, 'wsgi.run_once': False}
        environ.update(updates)
        result = {}

        def start(status, headers, exc_info=None):
            result.update(status=int(status.split()[0]), headers=dict(headers))
        response = validator(self.app)(environ, start)
        try:
            result['body'] = b''.join(response)
        finally:
            response.close()
        return result

    def test_sample_matches_saved_model_without_image_inference(self):
        with patch('foodvision.inference.Predictor.photos', side_effect=AssertionError('No uploads')):
            r = self.request('POST', '/api/predict', b'{"example_id":"L81"}', HTTP_ORIGIN='https://demo.example')
        self.assertEqual(r['status'], 200)
        self.assertAlmostEqual(json.loads(r['body'])['estimated_fraction'], 0.8013227315352864)

    def test_private_routes_and_traversal_are_absent(self):
        for route in ('/capture', '/lab', '/api/capture', '/api/capture/export', '/api/lab', '/api/highlight', '/web/lab.js', '/web/capture.html', '/data/records.json', '/models/demo_model.npz', '/web/../data/checksums.json', '/web/%2e%2e/data/records.json', '/.local/capture.sqlite3'):
            with self.subTest(route=route):
                self.assertEqual(self.request(path=route)['status'], 404)
                self.assertEqual(self.request('POST', route, b'{}')['status'], 404)

    def test_simultaneous_sample_requests_are_consistent(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.request('POST', '/api/predict', b'{"example_id":"L492"}'), range(32)))
        self.assertTrue(all((r['status'] == 200 for r in results)))
        self.assertEqual(len({r['body'] for r in results}), 1)
if __name__ == '__main__':
    unittest.main()
