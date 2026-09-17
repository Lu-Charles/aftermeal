"""Local weighed-serving collection with atomic photos, revisions and exports."""
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import uuid
import zipfile
from .inference import decode_image
from .input_checks import pixel_digest
MATERIALS = {'portion', 'residue', 'empty', 'uncertain'}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def text(value, label, limit=200, required=True):
    if not isinstance(value, str) or len(value.strip()) > limit or (required and (not value.strip())):
        raise ValueError(f"{label} must be {('nonempty ' if required else '')}text, up to {limit} characters.")
    return value.strip()

def mass(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('Enter scale readings in grams.')
    try:
        number = Decimal(str(value)) * 1000
        if not number.is_finite() or not 0 <= number <= 20000000 or number != number.to_integral_value():
            raise ValueError('Scale readings must be 0–20,000 grams, with at most three decimals.')
        return int(number)
    except InvalidOperation as exc:
        raise ValueError('Invalid scale reading.') from exc

class CaptureStore:

    def __init__(self, database):
        self.database = Path(database)

    @contextmanager
    def connection(self):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript('\n          CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL, created TEXT NOT NULL);\n          CREATE TABLE IF NOT EXISTS servings(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), food TEXT NOT NULL,\n            tare_mg INTEGER NOT NULL, resolution_mg INTEGER NOT NULL, created TEXT NOT NULL, exclusion TEXT);\n          CREATE TABLE IF NOT EXISTS photos(id TEXT PRIMARY KEY, serving_id TEXT NOT NULL REFERENCES servings(id),\n            raw_sha TEXT NOT NULL, pixels_sha TEXT NOT NULL UNIQUE, mime TEXT NOT NULL, bytes BLOB NOT NULL);\n          CREATE TABLE IF NOT EXISTS readings(id TEXT PRIMARY KEY, serving_id TEXT NOT NULL REFERENCES servings(id),\n            role TEXT NOT NULL, photo_id TEXT NOT NULL REFERENCES photos(id), created TEXT NOT NULL);\n          CREATE TABLE IF NOT EXISTS revisions(reading_id TEXT NOT NULL REFERENCES readings(id), revision INTEGER NOT NULL,\n            gross_mg INTEGER NOT NULL, material TEXT NOT NULL, note TEXT NOT NULL, created TEXT NOT NULL,\n            PRIMARY KEY(reading_id, revision));\n          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT NOT NULL);\n        ')
        try:
            yield db
        finally:
            db.close()

    def summary(self, db=None):
        if db is None:
            with self.connection() as connection:
                return self.summary(connection)
        sessions = [dict(r) for r in db.execute('SELECT * FROM sessions ORDER BY created,id')]
        for session in sessions:
            session['servings'] = [dict(r) for r in db.execute('SELECT * FROM servings WHERE session_id=? ORDER BY created,id', (session['id'],))]
            for serving in session['servings']:
                readings = [dict(r) for r in db.execute('SELECT * FROM readings WHERE serving_id=? ORDER BY created,id', (serving['id'],))]
                for reading in readings:
                    reading['history'] = [dict(r) for r in db.execute('SELECT revision,gross_mg,material,note,created FROM revisions WHERE reading_id=? ORDER BY revision', (reading['id'],))]
                    reading.update(reading['history'][-1])
                    photo = db.execute('SELECT raw_sha,pixels_sha,mime FROM photos WHERE id=?', (reading['photo_id'],)).fetchone()
                    reading['image'] = dict(photo)
                    reading['image']['url'] = '/api/capture/photos/' + reading['photo_id']
                    reading['net_mg'] = reading['gross_mg'] - serving['tare_mg']
                before = next((r for r in readings if r['role'] == 'before'))
                for reading in readings:
                    reading['remaining_fraction'] = reading['net_mg'] / before['net_mg']
                    reading['below_scale_resolution'] = reading['net_mg'] < serving['resolution_mg']
                serving['readings'] = readings
        return {'schema_version': 1, 'kind': 'user_recorded_scale_measurements', 'sessions': sessions, 'measurement_status': 'Not independently verified; excluded records must not be used for fitting or scoring.'}

    def mutate(self, data):
        if not isinstance(data, dict):
            raise ValueError('Expected a capture record.')
        request_id = text(data.get('request_id'), 'Request ID', 80)
        fingerprint = hashlib.sha256(canonical(data).encode()).hexdigest()
        action = data.get('action')
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db, db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT digest,result FROM requests WHERE id=?', (request_id,)).fetchone()
            if previous:
                if previous['digest'] != fingerprint:
                    raise ValueError('This retry ID belongs to a different record. Reload before saving.')
                return json.loads(previous['result'])
            new_id = uuid.uuid4().hex
            if action == 'session':
                name = text(data.get('name'), 'Session name')
                purpose = data.get('purpose')
                if purpose not in {'pilot', 'evaluation'}:
                    raise ValueError('Choose pilot or reserved evaluation.')
                db.execute('INSERT INTO sessions VALUES (?,?,?,?)', (new_id, name, purpose, now))
                result = {'session_id': new_id}
            elif action == 'serving':
                session_id = text(data.get('session_id'), 'Session ID', 80)
                if not db.execute('SELECT 1 FROM sessions WHERE id=?', (session_id,)).fetchone():
                    raise ValueError('Unknown session.')
                food = text(data.get('food'), 'Food description')
                tare, resolution, gross = (mass(data.get(k)) for k in ('tare_g', 'resolution_g', 'gross_g'))
                if not 1 <= resolution <= 10000:
                    raise ValueError('Scale increment must be between 0.001 and 10 grams.')
                if gross <= tare:
                    raise ValueError('Starting plate + food must weigh more than the empty plate.')
                if tare % resolution or gross % resolution:
                    raise ValueError('Readings must match the scale increment.')
                db.execute('INSERT INTO servings VALUES (?,?,?,?,?,?,NULL)', (new_id, session_id, food, tare, resolution, now))
                reading_id = self.add_reading(db, new_id, 'before', gross, 'portion', data, now)
                result = {'serving_id': new_id, 'reading_id': reading_id}
            elif action in {'reading', 'correction', 'exclude'}:
                serving_id = text(data.get('serving_id'), 'Serving ID', 80)
                serving = db.execute('SELECT * FROM servings WHERE id=?', (serving_id,)).fetchone()
                if not serving:
                    raise ValueError('Unknown serving.')
                if serving['exclusion']:
                    raise ValueError('This serving is excluded. Start a new serving.')
                if action == 'exclude':
                    reason = text(data.get('reason'), 'Exclusion reason', 1000)
                    db.execute('UPDATE servings SET exclusion=? WHERE id=?', (reason, serving_id))
                    result = {'serving_id': serving_id, 'excluded': True}
                else:
                    gross = mass(data.get('gross_g'))
                    if gross < serving['tare_mg']:
                        raise ValueError('Plate + food cannot weigh less than the empty plate.')
                    if gross % serving['resolution_mg']:
                        raise ValueError('Reading must match the scale increment.')
                    before = db.execute("SELECT v.gross_mg FROM readings r JOIN revisions v ON r.id=v.reading_id WHERE r.serving_id=? AND r.role='before' ORDER BY v.revision DESC LIMIT 1", (serving_id,)).fetchone()[0]
                    if gross > before:
                        raise ValueError('This exceeds the starting weight. Added food or a changed plate needs a new serving.')
                    material = data.get('material')
                    if material not in MATERIALS:
                        raise ValueError('Choose what is visible in the photo.')
                    if action == 'reading':
                        result = {'serving_id': serving_id, 'reading_id': self.add_reading(db, serving_id, 'after', gross, material, data, now)}
                    else:
                        reading_id = text(data.get('reading_id'), 'Reading ID', 80)
                        reading = db.execute('SELECT * FROM readings WHERE id=? AND serving_id=?', (reading_id, serving_id)).fetchone()
                        if not reading or reading['role'] != 'after':
                            raise ValueError('Only after readings can be corrected. Exclude a serving with an incorrect starting record.')
                        revision = db.execute('SELECT MAX(revision) FROM revisions WHERE reading_id=?', (reading_id,)).fetchone()[0]
                        if type(data.get('expected_revision')) is not int or data['expected_revision'] != revision:
                            raise ValueError('This record changed elsewhere. Reload before correcting it.')
                        note = text(data.get('note'), 'Correction reason', 1000)
                        db.execute('INSERT INTO revisions VALUES (?,?,?,?,?,?)', (reading_id, revision + 1, gross, material, note, now))
                        result = {'serving_id': serving_id, 'reading_id': reading_id, 'revision': revision + 1}
            else:
                raise ValueError('Unknown capture action.')
            db.execute('INSERT INTO requests VALUES (?,?,?)', (request_id, fingerprint, canonical(result)))
            return result

    def add_reading(self, db, serving_id, role, gross, material, data, now):
        photo_string = data.get('photo')
        if not isinstance(photo_string, str):
            raise ValueError('Choose a photo.')
        raw = base64.b64decode(photo_string, validate=True)
        photo = decode_image(raw)
        if min(photo.size) < 224:
            raise ValueError('Photo must be at least 224 × 224 pixels.')
        pixels = pixel_digest(photo)
        if db.execute('SELECT 1 FROM photos WHERE pixels_sha=?', (pixels,)).fetchone():
            raise ValueError('This photo is already recorded. Take a new photo for each weighing; retries do not need a new record.')
        photo_id, reading_id = (uuid.uuid4().hex, uuid.uuid4().hex)
        mime = 'image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg'
        db.execute('INSERT INTO photos VALUES (?,?,?,?,?,?)', (photo_id, serving_id, hashlib.sha256(raw).hexdigest(), pixels, mime, raw))
        db.execute('INSERT INTO readings VALUES (?,?,?,?,?)', (reading_id, serving_id, role, photo_id, now))
        db.execute('INSERT INTO revisions VALUES (?,?,?,?,?,?)', (reading_id, 1, gross, material, text(data.get('note', ''), 'Note', 1000, False), now))
        return reading_id

    def image(self, photo_id):
        with self.connection() as db:
            row = db.execute('SELECT bytes,mime FROM photos WHERE id=?', (photo_id,)).fetchone()
            if not row:
                raise FileNotFoundError('Unknown capture photo.')
            return (row['bytes'], row['mime'])
