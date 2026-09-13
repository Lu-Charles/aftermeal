import base64
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from PIL import Image
from foodvision.capture import CaptureStore, mass
from foodvision.server import make_handler

def photo(color):
    stream = io.BytesIO()
    Image.new('RGB', (224, 224), color).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode()

class CaptureChecks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = CaptureStore(Path(self.tmp.name) / 'capture.sqlite3')
        self.sid = self.save(action='session', name='Pilot', purpose='pilot')['session_id']

    def save(self, **payload):
        return self.store.mutate({'request_id': uuid.uuid4().hex, **payload})

    def start(self, **overrides):
        payload = dict(action='serving', session_id=self.sid, food='Rice', tare_g='250.1', gross_g='350.1', resolution_g='.1', photo=photo('red'))
        payload.update(overrides)
        return self.save(**payload)

    def after(self, serving_id, **overrides):
        payload = dict(action='reading', serving_id=serving_id, gross_g='275.1', material='residue', photo=photo('blue'))
        payload.update(overrides)
        return self.save(**payload)

    def test_subtracts_plate_and_preserves_residue_weight(self):
        s = self.start()
        self.after(s['serving_id'])
        reading = self.store.summary()['sessions'][0]['servings'][0]['readings'][1]
        self.assertEqual(reading['net_mg'], 25000)
        self.assertEqual(reading['remaining_fraction'], 0.25)
        self.assertEqual(reading['material'], 'residue')

    def test_retry_is_idempotent_but_changed_payload_is_rejected(self):
        data = dict(request_id='retry-1', action='session', name='Reserved', purpose='evaluation')
        self.assertEqual(self.store.mutate(data), self.store.mutate(data))
        with self.assertRaises(ValueError):
            self.store.mutate({**data, 'name': 'Changed'})
        self.assertEqual(len(self.store.summary()['sessions']), 2)

    def test_photo_failure_rolls_back_serving_and_retry_record(self):
        with self.assertRaises(ValueError):
            self.start(photo='bm90IGFuIGltYWdl')
        with self.store.connection() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM servings').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM requests').fetchone()[0], 1)

    def test_same_photo_cannot_cross_pilot_and_reserved_sessions(self):
        self.start()
        reserved = self.save(action='session', name='Test', purpose='evaluation')['session_id']
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            self.start(session_id=reserved)
        self.assertEqual(self.store.summary()['sessions'][1]['servings'], [])

    def test_invalid_scale_readings_are_rejected(self):
        for value in [True, 'NaN', 'Infinity', '-1', '1.0001', 20001]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                mass(value)
        for extra in [dict(gross_g='250.1'), dict(gross_g='350.15'), dict(resolution_g='0')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.start(**extra)
        self.assertEqual(mass('0.001'), 1)

    def test_after_must_use_same_tare_and_not_exceed_start(self):
        s = self.start()['serving_id']
        for gross in ['250', '351']:
            with self.assertRaises(ValueError):
                self.after(s, gross_g=gross)
        self.after(s, gross_g='250.1')
        r = self.store.summary()['sessions'][0]['servings'][0]['readings'][-1]
        self.assertTrue(r['below_scale_resolution'])
        self.assertEqual(r['remaining_fraction'], 0)

    def test_exclusion_retains_records_and_blocks_additions(self):
        s = self.start()['serving_id']
        self.save(action='exclude', serving_id=s, reason='Wrong starting plate')
        with self.assertRaisesRegex(ValueError, 'excluded'):
            self.after(s)
        self.assertEqual(self.store.summary()['sessions'][0]['servings'][0]['exclusion'], 'Wrong starting plate')

class CaptureHttpChecks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        handler = make_handler(capture_database=Path(self.tmp.name) / 'capture.sqlite3')
        handler.log_message = lambda *a: None
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, method='GET', data=None, headers=None):
        c = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        c.request(method, path, body=json.dumps(data) if data else None, headers=headers or {})
        r = c.getresponse()
        result = (r.status, r.read(), dict(r.getheaders()))
        c.close()
        return result

    def test_capture_rejects_external_origin_and_rebinding_host(self):
        self.assertEqual(self.request('/api/capture', headers={'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('/api/capture', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('/api/capture', 'POST', {'action': 'session'}, headers={'Origin': 'https://evil.example'})[0], 403)

    def test_capture_post_persists_and_invalid_photo_does_not(self):
        status, body, _ = self.request('/api/capture', 'POST', {'request_id': 'x', 'action': 'session', 'name': 'Kitchen', 'purpose': 'pilot'})
        self.assertEqual(status, 200)
        self.assertIn('session_id', json.loads(body))
        self.assertEqual(len(json.loads(self.request('/api/capture')[1])['sessions']), 1)
