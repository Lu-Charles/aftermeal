"""Public samples and optional bounded, in-memory photo inference."""
import base64
import binascii
from collections import deque
from http import HTTPStatus
import json
import logging
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlsplit
import threading
import time
from .benchmark import verify_payload
from .inference import Predictor, ROOT
VERSION = '1.1.0'
MAX_PUBLIC_BODY = 1024
MAX_UPLOAD_BODY = 12 * 1024 * 1024
WEB_FILES = ('index.html', 'app.js', 'style.css')
CSP = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

def capabilities(root=ROOT, *, public=False, hosted_uploads=False):
    manifest = json.loads((root / 'data/checksums.json').read_text())
    return {'version': VERSION, 'mode': ('hosted' if hosted_uploads else 'public') if public else 'local',
            'uploads': hosted_uploads or not public, 'upload_highlights': not public,
            'collection': not public, 'model_sha256': manifest['models/demo_model.npz'],
            'example_images': {p.removeprefix('examples/images/'): digest for p, digest in manifest.items() if p.startswith('examples/images/')}}

class PublicApplication:
    """Precompute four predictions at startup; each request serves bounded bytes."""

    def __init__(self, root: Path=ROOT, *, upload_predictor=None, clock=time.monotonic):
        verify_payload(root)
        self.routes = {}
        self.upload_predictor = upload_predictor
        self.upload_lock = threading.Lock()
        self.upload_times = deque()
        self.clock = clock
        predictor = Predictor(root)
        examples = json.loads((root / 'examples/examples.json').read_text())
        self.predictions = {item['id']: self.json_bytes(predictor.example(item['id'])) for item in examples}
        for name in WEB_FILES:
            path = root / 'web' / name
            self.routes['/web/' + name] = (path.read_bytes(), mimetypes.guess_type(name)[0] or 'text/javascript')
        self.routes['/'] = self.routes['/web/index.html']
        self.routes['/case-study'] = self.routes['/']
        self.routes['/web/case-study.html'] = self.routes['/']
        for item in examples:
            for kind in ('before', 'after'):
                for route in (item[kind], f"/examples/images/overlays/{item['id']}_{kind}.jpg"):
                    self.routes[route] = ((root / route.lstrip('/')).read_bytes(), 'image/jpeg')
        config = capabilities(root, public=True, hosted_uploads=upload_predictor is not None)
        for route, data in {'/api/examples': examples, '/api/config': config, '/api/model': {k: predictor.metadata[k] for k in ('feature', 'calibration_records', 'source')}, '/api/benchmark': json.loads((root / 'reports/benchmark.json').read_text()), '/healthz': {'status': 'ok', 'version': VERSION, 'mode': config['mode'], 'examples': len(examples), 'uploads': config['uploads']}}.items():
            self.routes[route] = (self.json_bytes(data), 'application/json')

    @staticmethod
    def json_bytes(value):
        return json.dumps(value, allow_nan=False).encode()

    def __call__(self, environ, start_response):
        method, path = (environ.get('REQUEST_METHOD', 'GET'), environ.get('PATH_INFO', '/'))

        def respond(code, data, content_type='application/json', extra=()):
            if not isinstance(data, bytes):
                data = self.json_bytes(data)
            headers = [('Content-Type', content_type), ('Content-Length', str(len(data))), ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'), ('Content-Security-Policy', CSP), ('Referrer-Policy', 'no-referrer'), ('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')]
            start_response(f'{code} {HTTPStatus(code).phrase}', headers + list(extra))
            return [b'' if method == 'HEAD' else data]
        if method in ('GET', 'HEAD'):
            if path in self.routes:
                return respond(200, *self.routes[path])
            return respond(404, {'error': 'Not found.'})
        upload = path == '/api/upload' and self.upload_predictor is not None
        if path != '/api/predict' and not upload:
            return respond(404, {'error': 'Not found.'})
        if method != 'POST':
            return respond(405, {'error': 'Use POST.'}, extra=(('Allow', 'POST'),))
        origin = environ.get('HTTP_ORIGIN')
        if origin:
            try:
                parsed = urlsplit(origin)
                same_host = parsed.netloc == environ.get('HTTP_HOST') and parsed.scheme in ('http', 'https')
                valid_origin = not parsed.path and (not parsed.query) and (not parsed.fragment) and (parsed.username is None)
            except ValueError:
                same_host = valid_origin = False
            if not same_host or not valid_origin:
                return respond(403, {'error': 'Use the demo on this site.'})
        if environ.get('CONTENT_TYPE', '').split(';')[0].strip() != 'application/json':
            return respond(415, {'error': 'Send application/json.'})
        try:
            length = int(environ.get('CONTENT_LENGTH') or '0')
        except ValueError:
            return respond(400, {'error': 'Invalid Content-Length.'})
        if upload:
            if not 0 < length <= MAX_UPLOAD_BODY:
                return respond(413, {'error': 'Photo pair exceeds the upload limit.'})
            # Admit only one upload before reading any body, leaving other threads
            # available for samples and health checks without multiplying memory.
            if not self.upload_lock.acquire(blocking=False):
                return respond(429, {'error': 'Another photo pair is being analyzed. Please retry shortly.'},
                               extra=(('Retry-After', '3'),))
            try:
                now = self.clock()
                while self.upload_times and now - self.upload_times[0] >= 60:
                    self.upload_times.popleft()
                if len(self.upload_times) >= 12:
                    return respond(429, {'error': 'Photo analysis is busy. Please try again in a minute.'},
                                   extra=(('Retry-After', '60'),))
                self.upload_times.append(now)
                raw = environ['wsgi.input'].read(length)
                if len(raw) != length:
                    raise ValueError('Incomplete photo upload.')
                data = json.loads(raw)
                if not isinstance(data, dict) or set(data) != {'before', 'after'}:
                    raise ValueError('Upload one before photo and one after photo.')
                from .hosted import MAX_UPLOAD_BYTES
                limit = 4 * ((MAX_UPLOAD_BYTES + 2) // 3)
                if any(not isinstance(data[k], str) or not 0 < len(data[k]) <= limit for k in ('before', 'after')):
                    raise ValueError('Each photo must be at most 4 MB after resizing.')
                photos = [base64.b64decode(data[k], validate=True) for k in ('before', 'after')]
                return respond(200, self.upload_predictor.photos(*photos))
            except (ValueError, UnicodeError, binascii.Error) as exc:
                return respond(400, {'error': str(exc) if not isinstance(exc, json.JSONDecodeError) else 'Invalid JSON upload.'})
            except Exception:
                # Do not log request contents or exceptions containing photo data.
                logging.error('Hosted photo inference failed')
                return respond(503, {'error': 'Photo analysis is temporarily unavailable. Please retry.'})
            finally:
                self.upload_lock.release()
        if not 0 < length <= MAX_PUBLIC_BODY:
            return respond(413, {'error': 'This endpoint accepts only a sample ID (up to 1 KB).'})
        try:
            raw = environ['wsgi.input'].read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request.')
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != {'example_id'} or (not isinstance(data['example_id'], str)):
                return respond(400, {'error': 'Choose a sample pair for this endpoint.'})
            prediction = self.predictions.get(data['example_id'])
            if prediction is None:
                return respond(400, {'error': 'Unknown sample pair.'})
            return respond(200, prediction)
        except (ValueError, UnicodeError):
            return respond(400, {'error': 'Invalid JSON request.'})
        except Exception:
            logging.exception('Public prediction request failed')
            return respond(500, {'error': 'Could not load this result. Please retry.'})

def create_app():
    predictor = None
    if os.environ.get('AFTERMEAL_HOSTED_UPLOADS') == '1':
        from .hosted import HostedPredictor
        predictor = HostedPredictor(encoder_directory=os.environ.get('AFTERMEAL_ENCODER_DIRECTORY'))
    return PublicApplication(upload_predictor=predictor)
