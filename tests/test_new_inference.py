"""Guard the new experiments' boundaries and the upload highlight endpoint."""
import base64
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch
import numpy as np
from foodvision.data_budget import eligible, expanded_episode
from foodvision.features import meal_representations
from foodvision.hurdle import forecast, predictions
from foodvision.models import CandidateRegressors
from foodvision.server import make_handler

class ExperimentChecks(unittest.TestCase):

    def test_expansion_purges_cross_role_images_and_group_links(self):
        rows = [{'record_id': str(i), 'source': 's', 'group': str(i), 'similarity_component': str(i)} for i in range(12)]
        rows[1]['group'] = '0'
        rows[2]['similarity_component'] = '0'
        hashes = {'0': {'before', 'after'}, '3': {'other', 'before'}}
        self.assertEqual(eligible(rows, list(range(12)), [0], hashes), list(range(4, 12)))
        episode = expanded_episode(rows, {'source': 's', 'name': 'test', 'test': [0]}, hashes)
        self.assertEqual(episode['train'], list(range(4, 12)))
        self.assertCountEqual([i for f in episode['inner'] for i in f['validation']], episode['train'])

    def test_hurdle_outer_labels_cannot_affect_forecasts_or_selection(self):
        rng = np.random.default_rng(80)
        x = meal_representations(rng.normal(size=(8, 5)), rng.normal(size=(8, 5)))
        y = np.array([0, 0.2, 0, 0.8, 0.5, 0, 0.3, 0.7])
        groups = np.array([str(i) for i in range(8)])
        rows = [{'record_id': str(i), 'group': str(i), 'similarity_component': str(i)} for i in range(8)]
        episode = {'train': list(range(6)), 'test': [6, 7], 'inner': [{'train': [j for j in range(6) if i != j], 'validation': [i]} for i in range(6)]}
        kernels = CandidateRegressors(x, y, groups).kernels
        a, selection_a = forecast(rows, kernels, y, groups, episode)
        y[6:] = [100, -100]
        b, selection_b = forecast(rows, kernels, y, groups, episode)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(selection_a, selection_b)

    def test_no_positive_training_labels_gives_zero_without_test_label_fallback(self):
        x = meal_representations(np.eye(3), np.eye(3))
        y = np.array([0, 0, 1.0])
        groups = np.array(['a', 'b', 'c'])
        kernels = CandidateRegressors(x, y, groups).kernels
        np.testing.assert_array_equal(predictions(kernels, y, groups, [0, 1], [2]), np.zeros((36, 1)))

class HighlightHttpChecks(unittest.TestCase):

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

    def post(self, payload, origin=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        connection.request('POST', '/api/highlight', json.dumps(payload), headers)
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        return (response.status, body)

    def test_route_passes_uploaded_bytes_and_never_calls_percentage_model(self):
        with patch('foodvision.server.highlight_photos', return_value={'status': 'highlighted'}) as highlight, patch('foodvision.server.Predictor.photos', side_effect=AssertionError('Separate operation')):
            status, body = self.post({'before': base64.b64encode(b'one').decode(), 'after': base64.b64encode(b'two').decode()})
        self.assertEqual(status, 200)
        self.assertEqual(highlight.call_args.args, (b'one', b'two'))
        self.assertNotIn('estimated_fraction', body)

    def test_malformed_images_and_outside_origin_are_rejected(self):
        self.assertEqual(self.post({'before': '?', 'after': '?'})[0], 400)
        self.assertEqual(self.post({}, 'https://elsewhere.example')[0], 403)

    def test_missing_optional_model_returns_unavailable_not_a_fake_overlay(self):
        with patch('foodvision.server.highlight_photos', side_effect=RuntimeError('Missing model')):
            status, body = self.post({'before': 'b25l', 'after': 'dHdv'})
        self.assertEqual(status, 503)
        self.assertNotIn('images', body)
import io
import unittest
from PIL import Image
from foodvision.inference import decode_image

class PhotoOrientationChecks(unittest.TestCase):

    def test_exif_orientation_is_applied(self):
        stream = io.BytesIO()
        photo = Image.new('RGB', (230, 250))
        exif = photo.getexif()
        exif[274] = 6
        photo.save(stream, format='JPEG', exif=exif)
        self.assertEqual(decode_image(stream.getvalue()).size, (250, 230))
