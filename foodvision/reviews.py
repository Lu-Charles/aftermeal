"""Import browser reviews into a local, append-only SQLite audit history.

Visual observations never replace source masses or become evaluation gold here.
The importer checks provenance, not the identity or correctness of the reviewer.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from .inference import ROOT
OBSERVATIONS = {'visible_food', 'residue_only', 'clean_empty', 'uncertain'}
BOOLEAN_FIELDS = {'pair_issue', 'starting_portion_unverified', 'source_labels_viewed', 'model_overlay_viewed', 'ai_proposal_viewed', 'mass_verified', 'evaluation_gold'}
RECORD_FIELDS = BOOLEAN_FIELDS | {'record_id', 'before_observation', 'observation', 'note', 'reviewed_at', 'reviewer', 'evidence_version', 'image_sha256'}
EXPORT_FIELDS = {'schema_version', 'kind', 'evidence_version', 'exported_at', 'records'}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Review timestamps must be timezone-aware ISO strings.')
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('Invalid review timestamp.') from exc
    if date.tzinfo is None:
        raise ValueError('Review timestamps must include a timezone.')
    return date.astimezone(timezone.utc).isoformat()

def validate(payload, root=ROOT):
    evidence = json.loads((Path(root) / 'reports/review-queue/review-queue.json').read_text())
    rows = {row['record_id']: row for row in evidence['rows']}
    if not isinstance(payload, dict) or set(payload) != EXPORT_FIELDS:
        raise ValueError('Expected an unmodified Review lab schema-2 export.')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 2:
        raise ValueError('Only schema-2 reviews with both photo observations are supported.')
    if payload['kind'] != 'visual_review_not_mass_ground_truth':
        raise ValueError('Only visual reviews can be imported.')
    if payload['evidence_version'] != evidence['proposal_sha256']:
        raise ValueError('Export refers to a different evidence version.')
    timestamp(payload['exported_at'])
    records = payload['records']
    if not isinstance(records, list) or not 1 <= len(records) <= len(rows):
        raise ValueError('Export must contain between 1 and 524 reviews.')
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError('Review fields differ from the schema-2 contract.')
        rid = record['record_id']
        if not isinstance(rid, str) or rid not in rows or rid in seen:
            raise ValueError('Unknown or repeated record ID in export.')
        seen.add(rid)
        expected = {kind: rows[rid]['images'][kind]['sha256'] for kind in ('before', 'after')}
        if record['image_sha256'] != expected or record['evidence_version'] != payload['evidence_version']:
            raise ValueError(f'{rid}: image hashes or evidence version do not match the review queue.')
        for key in ('before_observation', 'observation'):
            if not isinstance(record[key], str) or record[key] not in OBSERVATIONS:
                raise ValueError(f'{rid}: invalid {key}.')
        if any((type(record[key]) is not bool for key in BOOLEAN_FIELDS)):
            raise ValueError(f'{rid}: flags must be JSON booleans.')
        if record['mass_verified'] or record['evaluation_gold']:
            raise ValueError(f'{rid}: visual review cannot certify mass or evaluation gold.')
        if record['starting_portion_unverified'] != (record['before_observation'] != 'visible_food'):
            raise ValueError(f'{rid}: starting-portion flag contradicts the observation.')
        if not isinstance(record['note'], str) or len(record['note']) > 2000:
            raise ValueError(f'{rid}: note must be text of at most 2000 characters.')
        if not isinstance(record['reviewer'], str) or not 1 <= len(record['reviewer'].strip()) <= 200:
            raise ValueError(f'{rid}: reviewer description is missing or too long.')
        timestamp(record['reviewed_at'])
    return records

def initialize(connection):
    connection.execute('PRAGMA foreign_keys = ON')
    connection.executescript('\n        CREATE TABLE IF NOT EXISTS exports (\n            digest TEXT PRIMARY KEY, imported_at TEXT NOT NULL, payload TEXT NOT NULL\n        );\n        CREATE TABLE IF NOT EXISTS reviews (\n            revision INTEGER PRIMARY KEY, digest TEXT NOT NULL UNIQUE,\n            record_id TEXT NOT NULL, evidence_version TEXT NOT NULL,\n            reviewed_at TEXT NOT NULL, payload TEXT NOT NULL\n        );\n        CREATE TABLE IF NOT EXISTS export_reviews (\n            export_digest TEXT NOT NULL REFERENCES exports(digest),\n            review_digest TEXT NOT NULL REFERENCES reviews(digest),\n            PRIMARY KEY (export_digest, review_digest)\n        );\n        CREATE INDEX IF NOT EXISTS record_history ON reviews(record_id, reviewed_at);\n    ')

def import_reviews(source, database, root=ROOT):
    source, database = (Path(source), Path(database))
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Review export exceeds 4 MB.')
    payload = json.loads(source.read_text())
    records = validate(payload, root)
    serialized = canonical(payload)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=10)
    try:
        initialize(connection)
        with connection:
            connection.execute('BEGIN IMMEDIATE')
            if connection.execute('SELECT 1 FROM exports WHERE digest=?', (digest,)).fetchone():
                return {'status': 'already_imported', 'export_sha256': digest, 'new_revisions': 0}
            connection.execute('INSERT INTO exports VALUES (?, ?, ?)', (digest, datetime.now(timezone.utc).isoformat(), serialized))
            inserted = 0
            for record in records:
                encoded = canonical(record)
                review_digest = hashlib.sha256(encoded.encode()).hexdigest()
                cursor = connection.execute('INSERT OR IGNORE INTO reviews(digest,record_id,evidence_version,reviewed_at,payload) VALUES (?,?,?,?,?)', (review_digest, record['record_id'], record['evidence_version'], timestamp(record['reviewed_at']), encoded))
                inserted += cursor.rowcount
                connection.execute('INSERT INTO export_reviews VALUES (?, ?)', (digest, review_digest))
        return {'status': 'imported', 'export_sha256': digest, 'new_revisions': inserted, 'unchanged_reviews': len(records) - inserted, 'evaluation_gold': False}
    finally:
        connection.close()

def review_status(database):
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError('No review database yet. Import a Review lab export first.')
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        total = connection.execute('SELECT COUNT(*) FROM exports').fetchone()[0]
        revisions = connection.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        rows = connection.execute('\n            SELECT payload FROM (\n                SELECT payload, ROW_NUMBER() OVER (\n                    PARTITION BY record_id, evidence_version ORDER BY reviewed_at DESC, revision DESC\n                ) AS position FROM reviews\n            ) WHERE position=1\n        ').fetchall()
        latest = [json.loads(row[0]) for row in rows]
        return {'imports': total, 'revisions': revisions, 'reviewed_pairs': len(latest), 'starting_portion_unverified': sum((r['starting_portion_unverified'] for r in latest)), 'pair_issues': sum((r['pair_issue'] for r in latest)), 'reviewer_identity_verified': False, 'evaluation_gold_pairs': 0, 'source_weights_modified': False}
    finally:
        connection.close()
