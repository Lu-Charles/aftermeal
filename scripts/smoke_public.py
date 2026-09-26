"""Black-box release checks against a running public server (stdlib only)."""
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

def check(base):

    def request(path, data=None):
        req = Request(base.rstrip('/') + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=5) as response:
                return (response.status, response.read())
        except HTTPError as exc:
            return (exc.code, exc.read())
    status, body = request('/healthz')
    assert status == 200 and json.loads(body)['mode'] == 'public'
    assert request('/case-study')[0] == 200
    for record in ('L81', 'L133', 'L388', 'L492'):
        status, body = request('/api/predict', json.dumps({'example_id': record}).encode())
        assert status == 200 and 0 <= json.loads(body)['estimated_fraction'] <= 1.25
    for path in ('/api/capture/export', '/lab', '/web/capture.html', '/data/records.json'):
        assert request(path)[0] == 404, path
    assert request('/api/predict', b'x' * 1025)[0] == 413
    assert request('/api/predict', b'{"before":"photo","after":"photo"}')[0] == 400
    print('Public smoke passed: health, four samples, About compatibility route, private routes and request limits.')
if __name__ == '__main__':
    check(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080')
