"""Hosted uploads must preserve isolation and bound work before reading photos."""
import base64
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from foodvision.hosted import HostedPredictor, MAX_UPLOAD_BYTES, preprocess
from foodvision.public import PublicApplication, MAX_UPLOAD_BODY
import test_public


def pair():
    return json.dumps({'before': base64.b64encode(b'before').decode(),
                       'after': base64.b64encode(b'after').decode()}).encode()


class HostedUploadChecks(unittest.TestCase):
    request = test_public.PublicTests.request

    def setUp(self):
        self.now = 100.0
        self.predictor = SimpleNamespace(photos=Mock(return_value={
            'status': 'estimated', 'estimated_fraction': .3, 'photos_stored': False}))
        self.app = PublicApplication(upload_predictor=self.predictor, clock=lambda: self.now)

    def test_upload_reaches_model_with_actual_bytes(self):
        response = self.request('POST', '/api/upload', pair(), HTTP_ORIGIN='https://demo.example')
        self.assertEqual(response['status'], 200)
        self.predictor.photos.assert_called_once_with(b'before', b'after')
        self.assertFalse(json.loads(response['body'])['photos_stored'])
        config = json.loads(self.request(path='/api/config')['body'])
        self.assertTrue(config['uploads'])
        self.assertEqual(config['mode'], 'hosted')
        self.assertFalse(config['collection'])
        self.assertFalse(config['upload_highlights'])

    def test_busy_upload_does_not_read_body_or_block_health(self):
        class NoRead(io.BytesIO):
            def read(self, *args):
                raise AssertionError('Busy/oversize requests must not be read')
        self.app.upload_lock.acquire()
        try:
            response = self.request('POST', '/api/upload', CONTENT_LENGTH='100',
                                    **{'wsgi.input': NoRead()})
            self.assertEqual(response['status'], 429)
            self.assertEqual(self.request(path='/healthz')['status'], 200)
        finally:
            self.app.upload_lock.release()
        response = self.request('POST', '/api/upload', CONTENT_LENGTH=str(MAX_UPLOAD_BODY + 1),
                                **{'wsgi.input': NoRead()})
        self.assertEqual(response['status'], 413)
        self.predictor.photos.assert_not_called()

    def test_rate_window_expires_and_errors_release_capacity(self):
        for _ in range(12):
            self.assertEqual(self.request('POST', '/api/upload', pair())['status'], 200)
        self.assertEqual(self.request('POST', '/api/upload', pair())['status'], 429)
        self.now += 61
        self.predictor.photos.side_effect = RuntimeError('Unavailable')
        self.assertEqual(self.request('POST', '/api/upload', pair())['status'], 503)
        self.predictor.photos.side_effect = None
        self.assertEqual(self.request('POST', '/api/upload', pair())['status'], 200)

    def test_upload_contract_origin_and_private_routes(self):
        for payload in (b'[]', b'bad', b'{}', b'{"before":12,"after":"YQ=="}',
                        b'{"before":"?","after":"?"}', b'{"before":"YQ==","after":"YQ==","url":"https://example.com"}'):
            self.assertEqual(self.request('POST', '/api/upload', payload)['status'], 400)
        self.assertEqual(self.request('POST', '/api/upload', pair(), HTTP_ORIGIN='https://evil.example')['status'], 403)
        self.assertEqual(self.request('POST', '/api/upload', pair(), CONTENT_TYPE='text/plain')['status'], 415)
        for route in ('/capture', '/lab', '/api/capture', '/api/highlight', '/models/hosted/encoder.onnx'):
            self.assertEqual(self.request(path=route)['status'], 404)
        self.predictor.photos.assert_not_called()


class HostedImageChecks(unittest.TestCase):
    def setUp(self):
        self.predictor = HostedPredictor.__new__(HostedPredictor)
        self.predictor.flags = []
        self.predictor.session = Mock()

    def test_duplicate_and_large_images_never_reach_encoder(self):
        image = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (256, 1))).convert('RGB')
        stream = io.BytesIO(); image.save(stream, format='PNG')
        result = self.predictor.photos(stream.getvalue(), stream.getvalue())
        self.assertEqual(result['status'], 'rejected')
        for content in (b'x' * (MAX_UPLOAD_BYTES + 1), b'not a photo'):
            with self.assertRaises(ValueError):
                self.predictor.photos(content, stream.getvalue())
        large = io.BytesIO(); Image.new('RGB', (4001, 1000)).save(large, format='PNG')
        with self.assertRaisesRegex(ValueError, '4 megapixels'):
            self.predictor.photos(large.getvalue(), stream.getvalue())
        self.predictor.session.run.assert_not_called()

    def test_preprocessing_is_finite_fixed_shape_float32(self):
        tensor = preprocess(Image.new('RGB', (301, 257), (32, 64, 96)))
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
