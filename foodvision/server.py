"""Serve local analysis and explicit persistent weighed-data collection."""
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
from .inference import Predictor, ROOT
from .lab import ReviewLab
from .highlighting import highlight_photos
from .capture import CaptureStore
from .public import capabilities, CSP
MAX_BODY = 23 * 1024 * 1024

def make_handler(root: Path=ROOT, capture_database=None):
    predictor = Predictor(root)
    lab = ReviewLab(root)
    capture = CaptureStore(capture_database or root / '.local/capture.sqlite3')

    class Handler(BaseHTTPRequestHandler):

        def send(self, status, data, content_type='application/json'):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            if content_type == 'application/zip':
                self.send_header('Content-Disposition', 'attachment; filename="aftermeal-capture.zip"')
            self.send_header('Content-Security-Policy', CSP)
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = unquote(urlparse(self.path).path)
            if route == '/api/config':
                return self.send(200, capabilities(root))
            if route == '/healthz':
                return self.send(200, {'status': 'ok', 'mode': 'local'})
            if route.startswith('/api/capture'):
                if not self.capture_origin_allowed():
                    return self.send(403, {'error': 'Capture is available only from this local app.'})
                try:
                    if route == '/api/capture':
                        return self.send(200, capture.summary())
                    if route == '/api/capture/export':
                        return self.send(200, capture.export(), 'application/zip')
                    match = re.fullmatch('/api/capture/photos/([a-f0-9]{32})', route)
                    if match:
                        raw, mime = capture.image(match[1])
                        return self.send(200, raw, mime)
                    return self.send(404, {'error': 'Unknown capture resource.'})
                except FileNotFoundError:
                    return self.send(404, {'error': 'Capture photo not found.'})
            if route.startswith('/api/lab'):
                try:
                    if route == '/api/lab':
                        return self.send(200, lab.summary())
                    match = re.fullmatch('/api/lab/images/(L[0-9]+)/(before|after)(-overlay)?\\.jpg', route)
                    if match:
                        return self.send(200, lab.image(match[1], match[2], bool(match[3])), 'image/jpeg')
                    return self.send(404, {'error': 'Unknown review resource.'})
                except FileNotFoundError:
                    return self.send(404, {'error': 'Local review evidence is unavailable. See docs/EXPERIMENTS.md for setup.'})
                except ValueError as exc:
                    return self.send(400, {'error': str(exc)})
            if route == '/api/examples':
                return self.send(200, json.loads((root / 'examples/examples.json').read_text()))
            if route == '/api/benchmark':
                path = root / 'reports/benchmark.json'
                return self.send(200, json.loads(path.read_text())) if path.exists() else self.send(404, {'error': 'Run the benchmark command first.'})
            if route == '/api/model':
                return self.send(200, {k: predictor.metadata[k] for k in ('feature', 'calibration_records', 'source')})
            if route == '/':
                path = root / 'web/index.html'
            elif route == '/lab':
                path = root / 'web/lab.html'
            elif route == '/capture':
                path = root / 'web/capture.html'
            elif route in ('/case-study', '/web/case-study.html'):
                path = root / 'web/index.html'
            elif route.startswith('/web/') or route.startswith('/examples/images/'):
                path = (root / route.lstrip('/')).resolve()
                allowed = [(root / 'web').resolve(), (root / 'examples/images').resolve()]
                if not any((path.is_relative_to(directory) for directory in allowed)):
                    return self.send(404, {'error': 'Not found.'})
            else:
                return self.send(404, {'error': 'Not found.'})
            if not path.is_file():
                return self.send(404, {'error': 'Not found.'})
            return self.send(200, path.read_bytes(), mimetypes.guess_type(str(path))[0] or 'application/octet-stream')

        def capture_origin_allowed(self):
            host = self.headers.get('Host', '')
            try:
                local = urlparse('http://' + host).hostname in {'127.0.0.1', 'localhost', '::1'}
            except ValueError:
                return False
            origin = self.headers.get('Origin')
            return local and (not origin or origin == 'http://' + host)

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ('/api/predict', '/api/highlight', '/api/capture'):
                return self.send(404, {'error': 'Not found.'})
            if route == '/api/capture' and (not self.capture_origin_allowed()):
                return self.send(403, {'error': 'Capture is available only from this local app.'})
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.send(403, {'error': 'Requests must come from this local demo.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    return self.send(413, {'error': 'Choose two photos smaller than 8 MB each.'})
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected a photo pair or example ID.')
                if route == '/api/capture':
                    result = capture.mutate(data)
                elif route == '/api/highlight':
                    images = [base64.b64decode(data[key], validate=True) for key in ('before', 'after')]
                    result = highlight_photos(*images, root=root)
                elif 'example_id' in data:
                    result = predictor.example(data['example_id'])
                else:
                    images = [base64.b64decode(data[key], validate=True) for key in ('before', 'after')]
                    result = predictor.photos(*images, starting_portion=data.get('starting_portion'))
                return self.send(200, result)
            except (ValueError, KeyError, TypeError, binascii.Error) as exc:
                return self.send(400, {'error': str(exc)})
            except RuntimeError as exc:
                return self.send(503, {'error': str(exc)})
            except Exception:
                return self.send(500, {'error': 'Saving failed. Your form is still available; retry or check local storage.' if route == '/api/capture' else 'Image analysis failed. Check the terminal, model installation and network connection.'})
    return Handler

def serve(port: int):
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler())
    print(f'Open http://127.0.0.1:{server.server_port} — Ctrl+C to stop', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
