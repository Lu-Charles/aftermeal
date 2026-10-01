"""Ensure invalid or explicitly unsuitable inputs cannot reach expensive inference."""
import base64
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
import threading
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, PngImagePlugin
from foodvision.inference import Predictor, ROOT
from foodvision.input_checks import inspect_pair, pixel_digest
from foodvision.server import make_handler

def photo(reverse=False):
    pixels = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    if reverse:
        pixels = pixels.T
    return Image.fromarray(pixels).convert('RGB')

def encoded(image, comment=''):
    output = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('comment', comment)
    image.save(output, format='PNG', pnginfo=metadata)
    return output.getvalue()

class InputChecks(unittest.TestCase):

    def test_missing_and_uncertain_observations_allow_analysis_without_asserting_food_presence(self):
        for observation in (None, 'uncertain'):
            report = inspect_pair([photo(), photo(True)], observation)
            self.assertEqual(report['status'], 'ready')
            self.assertEqual(report['starting_portion']['value'], observation)
            self.assertFalse(report['automatic_food_detection'])
        self.assertEqual(inspect_pair([photo(), photo(True)])['starting_portion']['source'], 'not_provided')

    def test_same_pixels_different_file_metadata_are_rejected_without_encoder(self):
        first, second = (encoded(photo(), 'first'), encoded(photo(), 'second'))
        self.assertNotEqual(first, second)
        with patch('foodvision.inference.image_encoder', side_effect=AssertionError('Must not encode')):
            result = Predictor().photos(first, second, 'visible_food')
        self.assertEqual(result['status'], 'rejected')
        self.assertNotIn('estimated_fraction', result)
        self.assertIn('duplicate_photos', [i['code'] for i in result['input_checks']['issues']])

    def test_explicit_empty_start_does_not_encode(self):
        for observation in ('empty_or_residue',):
            with self.subTest(observation=observation), patch('foodvision.inference.image_encoder', side_effect=AssertionError('Must not encode')):
                result = Predictor().photos(encoded(photo()), encoded(photo(True)), observation)
                self.assertEqual(result['status'], 'needs_review')
                self.assertNotIn('estimated_fraction', result)

    def test_blank_and_small_images_are_not_called_empty_plates(self):
        for bad, code in ((Image.new('RGB', (256, 256), 'white'), 'constant_image'), (photo().resize((223, 256)), 'image_too_small')):
            report = inspect_pair([bad, photo(True)], 'visible_food')
            self.assertEqual(report['status'], 'rejected')
            self.assertIn(code, [i['code'] for i in report['issues']])
            self.assertFalse(report['automatic_food_detection'])

    def test_declaring_visible_food_cannot_override_source_flag(self):
        before = photo()
        flag = {'record_id': 'test-record', 'decoded_rgb_sha256': pixel_digest(before)}
        report = inspect_pair([before, photo(True)], 'visible_food', [flag])
        self.assertEqual(report['status'], 'needs_review')
        self.assertEqual(report['issues'][0]['code'], 'known_starting_image_flag')

    def test_ready_means_user_observation_not_automatic_visual_validation(self):
        report = inspect_pair([photo(), photo(True)], 'visible_food')
        self.assertEqual(report['status'], 'ready')
        self.assertEqual(report['starting_portion']['source'], 'user_observation')
        self.assertFalse(report['automatic_food_detection'])

    def test_unknown_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            inspect_pair([photo(), photo(True)], 'probably_good')

    def test_known_image_flag_is_traceable_to_unverified_source_evidence(self):
        registry = json.loads((ROOT / 'data/input_review_flags.json').read_text())
        self.assertEqual(registry['images'][0]['record_id'], 'L476')
        self.assertIn('not human adjudicated', registry['images'][0]['label_type'])

class InputHttpChecks(unittest.TestCase):

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

    def test_direct_api_without_observation_reaches_encoder(self):
        payload = {'before': base64.b64encode(encoded(photo())).decode(), 'after': base64.b64encode(encoded(photo(True))).decode()}
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        with patch('foodvision.inference.image_encoder', side_effect=RuntimeError('Encoder reached')) as encoder:
            connection.request('POST', '/api/predict', json.dumps(payload), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            body = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 503)
        self.assertEqual(body['error'], 'Encoder reached')
        encoder.assert_called_once()
if __name__ == '__main__':
    unittest.main()
