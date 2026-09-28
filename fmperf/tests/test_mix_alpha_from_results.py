"""Unit tests for mix_alpha_from_results.py (alpha calibration from a results archive).

The module under test lives in the repository root (next to experiment_automation.py), so the
repository root is added to sys.path before importing it, exactly like test_workload_mix.py.
Every fixture is a synthetic archive written into a temporary directory: the repository's own
results/ tree is never read nor written. Run with:
    pytest fmperf/tests/test_mix_alpha_from_results.py
or
    python fmperf/tests/test_mix_alpha_from_results.py
"""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mix_alpha_from_results as calibration
from workload_mix import parse_workload_mixes

# Columns of results.csv the calibration reads (a subset of the store_results.py header).
COLUMNS = [
    'EXPERIMENT_TYPE', 'MODEL_USED', 'MIN_INPUT_TOKENS', 'MAX_INPUT_TOKENS',
    'MIN_OUTPUT_TOKENS', 'MAX_OUTPUT_TOKENS', 'REQ_MIN', 'EVALUATION', 'DURATION', 'GPU_COUNT',
    'STAGE', 'TERMINATION_REASON', 'LARGEST_TRUE', 'SMALLEST_FALSE', 'FINISHED', 'WORKLOAD_MIX',
    'ADDITIVE',
]


def _row(**overrides) -> dict:
    """One results.csv row with the defaults of a finished MST iteration."""
    row = {
        'EXPERIMENT_TYPE': 'MST',
        'MODEL_USED': 'meta-llama/Llama-3.1-8B-Instruct',
        'MIN_INPUT_TOKENS': '', 'MAX_INPUT_TOKENS': '',
        'MIN_OUTPUT_TOKENS': '', 'MAX_OUTPUT_TOKENS': '',
        'REQ_MIN': '', 'EVALUATION': 'TRUE', 'DURATION': '1800s', 'GPU_COUNT': '1',
        'STAGE': '2', 'TERMINATION_REASON': '', 'LARGEST_TRUE': '', 'SMALLEST_FALSE': '',
        'FINISHED': 'FALSE', 'WORKLOAD_MIX': '', 'ADDITIVE': 'FALSE',
    }
    row.update(overrides)
    return row


def _interval(in_min, in_max, out_min, out_max) -> dict:
    """Token-interval columns of a non-additive results.csv row."""
    return {
        'MIN_INPUT_TOKENS': in_min,
        'MAX_INPUT_TOKENS': in_max,
        'MIN_OUTPUT_TOKENS': out_min,
        'MAX_OUTPUT_TOKENS': out_max,
    }


def _write_iteration(archive: Path, cell: str, stamp: str, *rows) -> Path:
    """Write <archive>/<cell>/<stamp>/results.csv with the given rows (one per iteration)."""
    folder = archive / cell / stamp
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'results.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


class ArchiveTestCase(unittest.TestCase):
    """Base class: a temporary results tree plus a default two-profile archive."""

    archive_name = 'Experiment_MST_2026-09-20_10-00-00'

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='most_mix_calibration_')
        self.addCleanup(temporary.cleanup)
        self.results_dir = Path(temporary.name)
        self.archive = self.results_dir / self.archive_name

    def build_archive(self) -> Path:
        """Light profile (MST 400) + heavy profile (MST 20) + one additive row to ignore."""
        _write_iteration(
            self.archive, '1-100_1-100', '2026-09-20_10-05-00',
            _row(**_interval(1, 100, 1, 100), REQ_MIN=200, STAGE=1, LARGEST_TRUE=200),
            _row(**_interval(1, 100, 1, 100), REQ_MIN=800, EVALUATION='FALSE', STAGE=1,
                 LARGEST_TRUE=200),
        )
        _write_iteration(
            self.archive, '1-100_1-100', '2026-09-20_10-20-00',
            _row(**_interval(1, 100, 1, 100), REQ_MIN=400, LARGEST_TRUE=300, SMALLEST_FALSE=600,
                 FINISHED='TRUE'),
        )
        _write_iteration(
            self.archive, '300-600_100-300', '2026-09-20_10-40-00',
            _row(**_interval(300, 600, 100, 300), REQ_MIN=20, LARGEST_TRUE=16, SMALLEST_FALSE=24,
                 FINISHED='TRUE'),
        )
        _write_iteration(
            self.archive, 'mix_1-100_1-100@0.5+300-600_100-300@0.5', '2026-09-20_11-00-00',
            _row(REQ_MIN=350, FINISHED='TRUE', ADDITIVE='TRUE',
                 WORKLOAD_MIX='[(1-100:1-100,0.5),(300-600:100-300,0.5)]'),
        )
        return self.archive

    def calibrate(self, profiles, cu_shares=None, archive_name=None) -> tuple:
        """Run the calibration on the sandbox archive; returns (payload, captured stdout)."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            payload = calibration.calibrate(
                archive_name or self.archive_name,
                profiles,
                cu_shares,
                self.results_dir,
            )
        return payload, buffer.getvalue()

    def run_main(self, argv) -> str:
        """Run main() with a patched argv and return the captured stdout."""
        buffer = io.StringIO()
        with mock.patch.object(sys, 'argv', ['mix_alpha_from_results.py', *argv]):
            with redirect_stdout(buffer):
                calibration.main()
        return buffer.getvalue()


class TestScanCells(ArchiveTestCase):
    def test_cells_are_keyed_by_token_interval(self):
        self.build_archive()
        cells, additive_rows = calibration.scan_cells(self.archive)
        self.assertEqual(sorted(cells), [(1, 100, 1, 100), (300, 600, 100, 300)])
        self.assertEqual(len(cells[(1, 100, 1, 100)]['records']), 3)
        self.assertEqual(additive_rows, 1)

    def test_legacy_folder_is_matched_through_the_interval_columns(self):
        _write_iteration(
            self.archive, '32_64', '2026-09-20_12-00-00',
            _row(**_interval(32, 32, 64, 64), REQ_MIN=1000, FINISHED='TRUE'),
        )
        cells, additive_rows = calibration.scan_cells(self.archive)
        self.assertEqual(sorted(cells), [(32, 32, 64, 64)])
        self.assertEqual(additive_rows, 0)

    def test_records_describe_the_row_they_came_from(self):
        self.build_archive()
        cells, _ = calibration.scan_cells(self.archive)
        record = cells[(1, 100, 1, 100)]['records'][-1]
        self.assertEqual(record['parent_dir'], '1-100_1-100')
        self.assertEqual(record['source'], str(Path('1-100_1-100') / '2026-09-20_10-20-00'))
        self.assertEqual(record['finished'], 'TRUE')
        self.assertEqual(record['evaluation'], 'TRUE')
        self.assertEqual(record['model'], 'meta-llama/Llama-3.1-8B-Instruct')
        self.assertFalse(record['additive'])

    def test_additive_rows_are_never_calibration_cells(self):
        _write_iteration(
            self.archive, 'mix_1-100_1-100@0.5+300-600_100-300@0.5', '2026-09-20_11-00-00',
            _row(REQ_MIN=350, FINISHED='TRUE', ADDITIVE='TRUE',
                 WORKLOAD_MIX='[(1-100:1-100,0.5),(300-600:100-300,0.5)]'),
        )
        cells, additive_rows = calibration.scan_cells(self.archive)
        self.assertEqual(cells, {})
        self.assertEqual(additive_rows, 1)

    def test_row_without_interval_is_ignored(self):
        _write_iteration(self.archive, '1-100_1-100', '2026-09-20_10-05-00', _row(REQ_MIN=200))
        cells, additive_rows = calibration.scan_cells(self.archive)
        self.assertEqual(cells, {})
        self.assertEqual(additive_rows, 0)


class TestSelectCalibrationValue(ArchiveTestCase):
    def select(self, *rows) -> tuple:
        """Scan one cell built from `rows` and return the value it calibrates to."""
        _write_iteration(self.archive, '1-100_1-100', '2026-09-20_10-05-00', *rows)
        cells, _ = calibration.scan_cells(self.archive)
        return calibration.select_calibration_value(cells[(1, 100, 1, 100)]['records'])

    def test_finished_row_wins_over_the_other_iterations(self):
        value, note, record = self.select(
            _row(**_interval(1, 100, 1, 100), REQ_MIN=200, LARGEST_TRUE=200),
            _row(**_interval(1, 100, 1, 100), REQ_MIN=800, EVALUATION='FALSE', LARGEST_TRUE=200),
            _row(**_interval(1, 100, 1, 100), REQ_MIN=400, LARGEST_TRUE=300,
                 SMALLEST_FALSE=600, FINISHED='TRUE'),
        )
        self.assertEqual(value, 400)
        self.assertEqual(note, calibration.FINISHED_ROW_NOTE)
        self.assertEqual(record['finished'], 'TRUE')
        self.assertEqual(record['req_min'], 400)

    def test_stage1_hard_limit_falls_back_to_largest_true(self):
        value, note, _ = self.select(
            _row(**_interval(1, 100, 1, 100), REQ_MIN=64, EVALUATION='FALSE', STAGE=1,
                 LARGEST_TRUE=48, FINISHED='TRUE',
                 TERMINATION_REASON='FAILED_STAGE1_ITERATION_LIMIT_EXCEEDED (ITERATION_HARD_LIMIT)'),
        )
        self.assertEqual(value, 48)
        self.assertIn('LARGEST_TRUE', note)

    def test_unfinished_cell_uses_the_largest_confirmed_true(self):
        value, note, _ = self.select(
            _row(**_interval(1, 100, 1, 100), REQ_MIN=256, LARGEST_TRUE=256),
            _row(**_interval(1, 100, 1, 100), REQ_MIN=384, EVALUATION='FALSE', LARGEST_TRUE=192),
        )
        self.assertEqual(value, 256)
        self.assertIn('largest confirmed TRUE', note)

    def test_sustainable_req_min_is_the_last_resort(self):
        value, note, _ = self.select(
            _row(**_interval(1, 100, 1, 100), REQ_MIN=100),
            _row(**_interval(1, 100, 1, 100), REQ_MIN=300),
            _row(**_interval(1, 100, 1, 100), REQ_MIN=600, EVALUATION='FALSE'),
        )
        self.assertEqual(value, 300)
        self.assertIn('highest sustainable REQ_MIN', note)

    def test_finished_row_without_a_verdict_flag_is_usable(self):
        value, note, _ = self.select(
            _row(**_interval(1, 100, 1, 100), REQ_MIN=250, EVALUATION='', FINISHED='TRUE'),
        )
        self.assertEqual(value, 250)
        self.assertIn('EVALUATION', note)

    def test_cell_without_a_usable_value_is_reported(self):
        value, note, record = self.select(
            _row(**_interval(1, 100, 1, 100), REQ_MIN='', LARGEST_TRUE='', EVALUATION='FALSE'),
        )
        self.assertIsNone(value)
        self.assertIsNone(record)
        self.assertIn('no usable REQ_MIN', note)


class TestCalibrate(ArchiveTestCase):
    def test_sigmas_scale_the_most_capable_profile_to_one(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,300-600:100-300')
        self.assertEqual([p['sigma'] for p in payload['profiles']], [1.0, 20.0])
        self.assertIn('sigma (CU per request, most capable calibrated profile = 1): 1, 20', output)

    def test_balanced_mix_matches_the_calculator_example(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,300-600:100-300')
        expected = '[(1-100:1-100,0.952381),(300-600:100-300,0.047619)]'
        self.assertEqual(payload['workload_mixes'], expected)
        self.assertIn(f'WORKLOAD_MIXES={expected}', output)
        self.assertEqual(payload['parent_dir'], 'mix_1-100_1-100@0.952381+300-600_100-300@0.047619')
        self.assertEqual([p['alpha'] for p in payload['profiles']], [0.952381, 0.047619])
        self.assertTrue(payload['balanced_alpha_cu'])
        self.assertEqual([p['achieved_alpha_cu'] for p in payload['profiles']], [0.5, 0.5])

    def test_printed_mix_reparses_without_deviation(self):
        self.build_archive()
        payload, _ = self.calibrate('1-100:1-100,300-600:100-300')
        mixes = parse_workload_mixes(payload['workload_mixes'])
        self.assertEqual(len(mixes), 1)
        self.assertEqual(mixes[0]['parent_dir'], payload['parent_dir'])
        parsed = [profile['alpha'] for profile in mixes[0]['profiles']]
        printed = [profile['alpha'] for profile in payload['profiles']]
        self.assertLess(max(abs(a - b) for a, b in zip(parsed, printed)), 1e-9)

    def test_cu_shares_change_the_alphas(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,300-600:100-300', cu_shares='0.75,0.25')
        self.assertEqual([p['alpha'] for p in payload['profiles']], [0.983607, 0.016393])
        self.assertFalse(payload['balanced_alpha_cu'])
        self.assertIn('alpha_CU from --cu-shares', output)

    def test_prediction_and_suggested_start_follow_the_cu_model(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,300-600:100-300')
        self.assertEqual(payload['predicted_requests_per_minute'], 210.0)
        self.assertEqual(payload['suggested_req_min_start'], 210)
        self.assertIn('Suggested REQ_MIN_START=210', output)

    def test_profiles_are_matched_by_interval_not_by_folder_name(self):
        _write_iteration(
            self.archive, '32_64', '2026-09-20_12-00-00',
            _row(**_interval(32, 32, 64, 64), REQ_MIN=1000, FINISHED='TRUE'),
        )
        _write_iteration(
            self.archive, '300-600_100-300', '2026-09-20_10-40-00',
            _row(**_interval(300, 600, 100, 300), REQ_MIN=20, FINISHED='TRUE'),
        )
        payload, _ = self.calibrate('32:64,300-600:100-300')
        self.assertEqual(
            [p['label'] for p in payload['profiles']], ['32-32:64-64', '300-600:100-300']
        )
        self.assertEqual([p['req_min'] for p in payload['profiles']], [1000.0, 20.0])
        self.assertEqual([p['sigma'] for p in payload['profiles']], [1.0, 50.0])

    def test_additive_rows_of_a_normal_archive_are_ignored(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,300-600:100-300')
        self.assertIn('Warning: ignored 1 additive (WORKLOAD_MIX) row(s)', output)
        self.assertEqual(payload['profiles'][0]['req_min'], 400.0)

    def test_additive_archive_is_rejected(self):
        _write_iteration(
            self.archive, 'mix_1-100_1-100@0.5+300-600_100-300@0.5', '2026-09-20_11-00-00',
            _row(REQ_MIN=350, FINISHED='TRUE', ADDITIVE='TRUE',
                 WORKLOAD_MIX='[(1-100:1-100,0.5),(300-600:100-300,0.5)]'),
        )
        with self.assertRaises(SystemExit) as context:
            self.calibrate('1-100:1-100,300-600:100-300')
        self.assertIn('additive (WORKLOAD_MIX)', str(context.exception))

    def test_unknown_profile_is_skipped_and_the_cells_are_listed(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,700-800:900-1000')
        self.assertEqual([p['label'] for p in payload['profiles']], ['1-100:1-100'])
        self.assertIn('Warning: skipping profile "700-800:900-1000"', output)
        self.assertIn('available cells in the archive: 1-100:1-100, 300-600:100-300', output)
        self.assertEqual(payload['skipped_profiles'], [{
            'label': '700-800:900-1000',
            'reason': 'the archive has no cell with this interval',
        }])

    def test_unparsable_profile_label_is_skipped(self):
        self.build_archive()
        payload, output = self.calibrate('not-a-profile,300-600:100-300')
        self.assertEqual([p['label'] for p in payload['profiles']], ['300-600:100-300'])
        self.assertIn('not a valid inMin-inMax:outMin-outMax profile label', output)

    def test_no_calibratable_profile_raises(self):
        self.build_archive()
        with self.assertRaises(SystemExit) as context:
            self.calibrate('700-800:900-1000')
        self.assertIn('none of the requested profiles could be calibrated', str(context.exception))

    def test_single_profile_warns(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100')
        self.assertEqual(payload['workload_mixes'], '[(1-100:1-100,1)]')
        self.assertIn('Warning: only one profile', output)

    def test_cu_shares_count_mismatch_raises(self):
        self.build_archive()
        with self.assertRaises(SystemExit) as context:
            self.calibrate('1-100:1-100,300-600:100-300', cu_shares='0.75')
        self.assertIn('alpha_CU value(s) in --cu-shares', str(context.exception))

    def test_negative_cu_share_raises(self):
        self.build_archive()
        with self.assertRaises(SystemExit) as context:
            self.calibrate('1-100:1-100,300-600:100-300', cu_shares='1.5,-0.5')
        self.assertIn('must be >= 0', str(context.exception))

    def test_archive_can_be_found_by_a_partial_name(self):
        self.build_archive()
        payload, output = self.calibrate('1-100:1-100,300-600:100-300',
                                         archive_name='2026-09-20')
        self.assertEqual(payload['profiles'][0]['req_min'], 400.0)
        self.assertIn(self.archive_name, output)

    def test_several_matches_use_the_newest_and_warn(self):
        self.build_archive()
        newer = self.results_dir / 'Experiment_MST_2026-09-21_10-00-00'
        _write_iteration(
            newer, '1-100_1-100', '2026-09-21_10-05-00',
            _row(**_interval(1, 100, 1, 100), REQ_MIN=60, FINISHED='TRUE'),
        )
        payload, output = self.calibrate('1-100:1-100,300-600:100-300',
                                         archive_name='Experiment_MST*')
        self.assertIn('archives match', output)
        self.assertEqual(payload['profiles'][0]['req_min'], 60.0)

    def test_missing_archive_lists_the_available_ones(self):
        self.build_archive()
        with self.assertRaises(SystemExit) as context:
            self.calibrate('1-100:1-100', archive_name='Experiment_Nope')
        message = str(context.exception)
        self.assertIn('no results archive matches', message)
        self.assertIn(self.archive_name, message)


class TestMain(ArchiveTestCase):
    def main_argv(self, *extra) -> list:
        return [
            '--results', self.archive_name,
            '-p', '1-100:1-100,300-600:100-300',
            '--results-dir', str(self.results_dir),
            *extra,
        ]

    def test_main_prints_the_pasteable_line(self):
        self.build_archive()
        output = self.run_main(self.main_argv())
        self.assertIn('WORKLOAD_MIXES=[(1-100:1-100,0.952381),(300-600:100-300,0.047619)]', output)

    def test_main_json_is_parseable(self):
        self.build_archive()
        output = self.run_main(self.main_argv('--json'))
        payload = json.loads(output[output.index('\n{'):])
        self.assertEqual(payload['k'], 2)
        self.assertEqual(payload['suggested_req_min_start'], 210)
        self.assertEqual([p['alpha'] for p in payload['profiles']], [0.952381, 0.047619])

    def test_main_only_reads(self):
        """The calibration must never write: not into results/ and not into .env."""
        self.build_archive()
        before = sorted(str(path) for path in self.results_dir.rglob('*'))
        env_before = (REPO_ROOT / '.env').exists()
        self.run_main(self.main_argv())
        self.assertEqual(before, sorted(str(path) for path in self.results_dir.rglob('*')))
        self.assertEqual(env_before, (REPO_ROOT / '.env').exists())


if __name__ == '__main__':
    unittest.main()
