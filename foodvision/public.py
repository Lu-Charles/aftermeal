"""Read-only public demo. No image upload, database, model download or research paths."""
from http import HTTPStatus
import json
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit
from .benchmark import verify_payload
from .inference import Predictor, ROOT
VERSION = '1.0.2'
MAX_PUBLIC_BODY = 1024
WEB_FILES = ('index.html', 'app.js', 'style.css')
CSP = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

class PublicApplication:
    """Precompute four predictions at startup; each request serves bounded bytes."""

    def __init__(self, root: Path=ROOT):
        verify_payload(root)
        self.routes = {}
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
        for route, data in {'/api/examples': examples, '/api/model': {k: predictor.metadata[k] for k in ('feature', 'calibration_records', 'source')}, '/api/benchmark': json.loads((root / 'reports/benchmark.json').read_text())}.items():
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
        if path != '/api/predict':
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
        if not 0 < length <= MAX_PUBLIC_BODY:
            return respond(413, {'error': 'This demo accepts only a sample ID (up to 1 KB). Run locally to analyze your photos.'})
        try:
            raw = environ['wsgi.input'].read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request.')
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != {'example_id'} or (not isinstance(data['example_id'], str)):
                return respond(400, {'error': 'Choose a sample pair. Photo uploads are available in the local app.'})
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
    return PublicApplication()
