"""Protect diagnostic denominators and the distinction between zero and small."""
import tempfile
import unittest
from pathlib import Path
from foodvision.diagnostics import run, summarize

class DiagnosticChecks(unittest.TestCase):

    def test_zero_small_and_large_boundaries_are_disjoint(self):
        report = summarize([0, 0.0001, 0.1, 0.1001], [0, 0, 0.1, 0.1], ['a', 'a', 'b', 'b'])
        self.assertEqual([report[k]['records'] for k in report], [4, 1, 2, 1])
        self.assertEqual(report['positive_up_to_10_percent']['predictions_at_or_below_1_percent_count'], 1)

    def test_group_and_row_errors_have_different_denominators(self):
        all_rows = summarize([0, 0, 0], [1, 0, 0], ['small', 'large', 'large'])['all']
        self.assertAlmostEqual(all_rows['group_mae_pp'], 50)
        self.assertAlmostEqual(all_rows['row_mae_pp'], 100 / 3)
        self.assertEqual(all_rows['predictions_above_10_percent_count'], 1)

    def test_absent_slice_is_not_reported_as_perfect_accuracy(self):
        empty = summarize([0.5], [0.5], ['a'])['recorded_zero']
        self.assertEqual(empty['records'], 0)
        self.assertIsNone(empty['group_mae_pp'])
        self.assertIsNone(empty['row_p90_absolute_error_pp'])

    def test_invalid_inputs_are_rejected(self):
        for args in [([0], [float('nan')], ['a']), ([0], [0, 1], ['a'])]:
            with self.assertRaises(ValueError):
                summarize(*args)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text('existing report')
            with self.assertRaises(FileExistsError):
                run(path)
            self.assertEqual(path.read_text(), 'existing report')
