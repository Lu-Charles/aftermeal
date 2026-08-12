"""Review imports preserve provenance and survive retries without losing history."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from foodvision.inference import ROOT
from foodvision.reviews import import_reviews, initialize, review_status

class ReviewImportTests(unittest.TestCase):

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.database = self.directory / 'reviews.sqlite3'
        queue = json.loads((ROOT / 'reports/review-queue/review-queue.json').read_text())
        records = []
        for row in queue['rows'][:2]:
            records.append({'record_id': row['record_id'], 'before_observation': 'visible_food', 'observation': 'uncertain', 'pair_issue': False, 'starting_portion_unverified': False, 'note': 'Synthetic importer test, not a real annotation', 'reviewed_at': '2000-01-01T10:00:00Z', 'reviewer': 'automated test fixture', 'source_labels_viewed': False, 'model_overlay_viewed': False, 'ai_proposal_viewed': False, 'evidence_version': queue['proposal_sha256'], 'image_sha256': {k: row['images'][k]['sha256'] for k in ('before', 'after')}, 'mass_verified': False, 'evaluation_gold': False})
        self.payload = {'schema_version': 2, 'kind': 'visual_review_not_mass_ground_truth', 'evidence_version': queue['proposal_sha256'], 'exported_at': '2000-01-01T11:00:00Z', 'records': records}

    def save(self, payload=None, name='export.json'):
        path = self.directory / name
        path.write_text(json.dumps(payload or self.payload))
        return path

    def test_retry_and_reexport_do_not_duplicate_reviews(self):
        source = self.save()
        self.assertEqual(import_reviews(source, self.database)['new_revisions'], 2)
        self.assertEqual(import_reviews(source, self.database)['status'], 'already_imported')
        payload = deepcopy(self.payload)
        payload['exported_at'] = '2000-01-01T12:00:00Z'
        result = import_reviews(self.save(payload), self.database)
        self.assertEqual(result['new_revisions'], 0)
        self.assertEqual(result['unchanged_reviews'], 2)
        self.assertEqual(review_status(self.database)['revisions'], 2)

    def test_concurrent_retry_commits_once(self):
        source = self.save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: import_reviews(source, self.database), range(2)))
        self.assertCountEqual([r['status'] for r in results], ['imported', 'already_imported'])
        self.assertEqual(review_status(self.database)['imports'], 1)

    def test_revisions_are_retained_and_old_backup_cannot_replace_new_review(self):
        new = deepcopy(self.payload)
        new['records'][0].update(before_observation='clean_empty', starting_portion_unverified=True, reviewed_at='2000-01-01T12:00:00Z')
        import_reviews(self.save(new), self.database)
        import_reviews(self.save(), self.database)
        status = review_status(self.database)
        self.assertEqual(status['revisions'], 3)
        self.assertEqual(status['reviewed_pairs'], 2)
        self.assertEqual(status['starting_portion_unverified'], 1)
        self.assertEqual(status['evaluation_gold_pairs'], 0)
        self.assertFalse(status['source_weights_modified'])

    def test_invalid_evidence_or_certification_rejects_whole_batch(self):
        import_reviews(self.save(), self.database)
        cases = [('image_sha256', {'before': 'wrong', 'after': 'wrong'}), ('evidence_version', 'stale'), ('evaluation_gold', True), ('mass_verified', True), ('pair_issue', 1), ('before_observation', 'maybe'), ('starting_portion_unverified', True), ('reviewed_at', '2000-01-01T12:00:00')]
        for key, value in cases:
            with self.subTest(key=key):
                payload = deepcopy(self.payload)
                payload['records'][0]['note'] = 'A valid change before the invalid record'
                payload['records'][1][key] = value
                with self.assertRaises(ValueError):
                    import_reviews(self.save(payload), self.database)
                self.assertEqual(review_status(self.database)['revisions'], 2)

    def test_storage_failure_rolls_back_the_entire_import(self):
        with sqlite3.connect(self.database) as connection:
            initialize(connection)
            connection.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON reviews\n                WHEN (SELECT COUNT(*) FROM reviews) > 0\n                BEGIN SELECT RAISE(ABORT, 'test disk/write failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            import_reviews(self.save(), self.database)
        self.assertEqual(review_status(self.database)['imports'], 0)
        self.assertEqual(review_status(self.database)['revisions'], 0)

    def test_duplicate_ids_and_missing_before_observation_are_not_silently_accepted(self):
        duplicate = deepcopy(self.payload)
        duplicate['records'][1] = deepcopy(duplicate['records'][0])
        missing = deepcopy(self.payload)
        del missing['records'][0]['before_observation']
        for payload in (duplicate, missing):
            with self.assertRaises(ValueError):
                import_reviews(self.save(payload), self.database)
        self.assertFalse(self.database.exists())
if __name__ == '__main__':
    unittest.main()
