"""Unit tests for MergeResultsCsv.py (merged single-file results.csv download).

The module under test lives in fmperf/utils next to GpuCount.py and is invoked as a plain script
(`python fmperf/utils/MergeResultsCsv.py merge --root ... --output ...`) by the MoST API, so its
directory is added to sys.path instead of importing the fmperf package (which needs the cluster
dependencies). Every fixture is a synthetic results scope written into a temporary directory: the
repository's own results/ tree is never read nor written. Run with:
    pytest fmperf/tests/test_merge_results_csv.py
or
    python fmperf/tests/test_merge_results_csv.py
"""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parents[1] / 'utils'
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import MergeResultsCsv as merged

# Columns of results.csv the merge has to preserve: the header of an older iteration (BASE_COLUMNS,
# with a quote/comma heavy JSON field to prove the CSV round-trip) and of a newer one that also
# stored the GPU count and the stage.
BASE_COLUMNS = [
    'EXPERIMENT_TYPE', 'MODEL_USED', 'MIN_INPUT_TOKENS', 'MAX_INPUT_TOKENS',
    'MIN_OUTPUT_TOKENS', 'MAX_OUTPUT_TOKENS', 'REQ_MIN', 'EVALUATION', 'DURATION',
    'INPUT_TOKEN_PERCENTILES', 'OUTPUT_TOKEN_PERCENTILES', 'LARGEST_TRUE', 'SMALLEST_FALSE',
    'FINISHED',
]
EXTRA_COLUMNS = ['GPU_COUNT', 'STAGE']

PERCENTILES = '{"p50":18.0,"p75":35.0,"p95":60.0}'


def _row(**overrides) -> dict:
    """One results.csv row with the defaults of a finished MST iteration."""
    row = {
        'EXPERIMENT_TYPE': 'MST',
        'MODEL_USED': 'meta-llama/Llama-3.1-8B-Instruct',
        'MIN_INPUT_TOKENS': '1',
        'MAX_INPUT_TOKENS': '100',
        'MIN_OUTPUT_TOKENS': '1',
        'MAX_OUTPUT_TOKENS': '100',
        'REQ_MIN': '400',
        'EVALUATION': 'TRUE',
        'DURATION': '1800s',
        'INPUT_TOKEN_PERCENTILES': PERCENTILES,
        'OUTPUT_TOKEN_PERCENTILES': PERCENTILES,
        'LARGEST_TRUE': '',
        'SMALLEST_FALSE': '',
        'FINISHED': 'TRUE',
    }
    row.update(overrides)
    return row


def _write_iteration(scope: Path, experiment: str, stamp: str, columns, *rows) -> Path:
    """Write <scope>/<experiment>/<stamp>/results.csv with the given columns and rows."""
    folder = scope / experiment / stamp
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'results.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


class MergeResultsCsvTestCase(unittest.TestCase):
    """Base class: a temporary results scope with two matrix cells and one additive folder."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix='most_merged_csv_')
        self.addCleanup(temporary.cleanup)
        self.temp_dir = Path(temporary.name)
        self.scope = self.temp_dir / 'results'

    def build_scope(self) -> None:
        """1-100_1-100 (two iterations, older header) + 1-100_300-600 (newer header) + additive."""
        _write_iteration(
            self.scope, '1-100_1-100', '2026-09-20_10-05-00', BASE_COLUMNS, _row(REQ_MIN='200')
        )
        _write_iteration(
            self.scope, '1-100_1-100', '2026-09-20_10-00-00', BASE_COLUMNS, _row(REQ_MIN='100')
        )
        _write_iteration(
            self.scope, '1-100_300-600', '2026-09-20_10-07-00', BASE_COLUMNS + EXTRA_COLUMNS,
            _row(MIN_OUTPUT_TOKENS='100', MAX_OUTPUT_TOKENS='300', REQ_MIN='25',
                 GPU_COUNT='1', STAGE='1'),
        )
        _write_iteration(
            self.scope, 'mix_1-100_1-100@1', '2026-09-20_11-00-00', BASE_COLUMNS,
            _row(EXPERIMENT_TYPE='ADDITIVE', REQ_MIN='400'),
        )

    def merge(self, **kwargs):
        """Merge the temporary scope and return ``(columns, rows, summary)``."""
        return merged.merge_results_csv(self.scope, **kwargs)

    def read_csv(self, path: Path):
        """Read a written merged CSV back into a list of dicts."""
        with path.open('r', encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    def test_merge_keeps_identity_and_orders_iterations(self):
        self.build_scope()
        columns, rows, summary = self.merge()
        self.assertEqual(columns[:2], ['IDENTIFIER', 'DATE'])
        self.assertEqual(summary['rows'], 3)
        self.assertEqual(summary['iterations'], 3)
        self.assertEqual(summary['skipped'], [])
        self.assertEqual(summary['experiments'], ['1-100_1-100', '1-100_300-600'])
        self.assertEqual(
            [row['IDENTIFIER'] for row in rows],
            ['1-100_1-100', '1-100_1-100', '1-100_300-600'],
        )
        # Iterations are merged chronologically, not in filesystem listing order.
        self.assertEqual(
            [row['DATE'] for row in rows],
            ['2026-09-20_10-00-00', '2026-09-20_10-05-00', '2026-09-20_10-07-00'],
        )
        self.assertEqual([row['REQ_MIN'] for row in rows], ['100', '200', '25'])

    def test_header_unions_the_source_columns_in_first_seen_order(self):
        self.build_scope()
        columns, rows, _ = self.merge()
        self.assertEqual(columns[2:], BASE_COLUMNS + EXTRA_COLUMNS)
        # The cell written with the older header keeps empty values for the newer columns.
        self.assertEqual(rows[0]['GPU_COUNT'], '')
        self.assertEqual(rows[0]['STAGE'], '')
        self.assertEqual(rows[2]['GPU_COUNT'], '1')
        self.assertEqual(rows[2]['STAGE'], '1')

    def test_date_column_value_takes_precedence_over_the_folder_name(self):
        _write_iteration(
            self.scope, '1-100_1-100', '2026-09-20_10-00-00', BASE_COLUMNS + ['Date', 'timestamp'],
            _row(Date='2026-09-20T10:00:00Z', timestamp='2026-09-20T10:00:00Z'),
        )
        columns, rows, _ = self.merge()
        self.assertEqual(rows[0]['DATE'], '2026-09-20T10:00:00Z')
        # The source date columns are still exported next to the merged DATE column.
        self.assertIn('Date', columns[2:])
        self.assertIn('timestamp', columns[2:])

    def test_additive_folders_are_skipped_by_default(self):
        self.build_scope()
        _, rows, summary = self.merge()
        self.assertNotIn('mix_1-100_1-100@1', [row['IDENTIFIER'] for row in rows])
        self.assertNotIn('mix_1-100_1-100@1', summary['experiments'])

    def test_include_additive_merges_the_mix_folders(self):
        self.build_scope()
        _, rows, summary = self.merge(include_additive=True)
        self.assertEqual([row['IDENTIFIER'] for row in rows][-1], 'mix_1-100_1-100@1')
        self.assertEqual(summary['experiments'][-1], 'mix_1-100_1-100@1')
        self.assertEqual(summary['rows'], 4)

    def test_explicit_additive_experiment_is_merged_without_the_flag(self):
        self.build_scope()
        _, rows, summary = self.merge(experiments='mix_1-100_1-100@1')
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary['experiments'], ['mix_1-100_1-100@1'])
        self.assertEqual(rows[0]['EXPERIMENT_TYPE'], 'ADDITIVE')

    def test_experiments_filter_limits_and_orders_the_rows(self):
        self.build_scope()
        _, rows, summary = self.merge(experiments='1-100_300-600, 1-100_1-100')
        self.assertEqual(
            [row['IDENTIFIER'] for row in rows],
            ['1-100_300-600', '1-100_1-100', '1-100_1-100'],
        )
        self.assertEqual(summary['experiments'], ['1-100_300-600', '1-100_1-100'])

    def test_unknown_experiment_is_reported_as_skipped(self):
        self.build_scope()
        _, rows, summary = self.merge(experiments='1-100_1-100,nope')
        self.assertEqual(summary['rows'], 2)
        self.assertEqual(summary['skipped'], ['nope: experiment folder not found'])
        self.assertEqual([row['IDENTIFIER'] for row in rows], ['1-100_1-100', '1-100_1-100'])

    def test_multiple_rows_per_iteration_are_all_merged(self):
        self.scope.mkdir(parents=True, exist_ok=True)
        _write_iteration(
            self.scope, '1-100_1-100', '2026-09-20_10-00-00', BASE_COLUMNS,
            _row(REQ_MIN='100'), _row(REQ_MIN='200'),
        )
        _, rows, _ = self.merge()
        self.assertEqual([row['REQ_MIN'] for row in rows], ['100', '200'])

    def test_written_csv_round_trips_the_quoted_fields(self):
        self.build_scope()
        columns, rows, _ = self.merge()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            destination = merged.write_merged_csv(columns, rows, '-')
        self.assertEqual(destination, '<stdout>')
        parsed = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]['INPUT_TOKEN_PERCENTILES'], PERCENTILES)
        self.assertEqual(parsed[0]['DATE'], '2026-09-20_10-00-00')

    def test_missing_root_raises_root_not_found(self):
        with self.assertRaises(merged.MergeResultsError) as context:
            self.merge()
        self.assertEqual(context.exception.code, 'ROOT_NOT_FOUND')
        self.assertIn('does not exist', context.exception.message)

    def test_scope_without_results_raises_no_results_found(self):
        # Scope directory without a single experiment folder.
        self.scope.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(merged.MergeResultsError) as context:
            self.merge()
        self.assertEqual(context.exception.code, 'NO_RESULTS_FOUND')

        # An experiment folder without any iteration results.csv is just as empty.
        (self.scope / '1-100_1-100' / '2026-09-20_10-00-00').mkdir(parents=True)
        with self.assertRaises(merged.MergeResultsError) as context:
            self.merge()
        self.assertEqual(context.exception.code, 'NO_RESULTS_FOUND')

    def test_unwritable_output_raises_write_failed(self):
        self.build_scope()
        columns, rows, _ = self.merge()
        blocker = self.temp_dir / 'blocker'
        blocker.write_text('not a directory', encoding='utf-8')
        with self.assertRaises(merged.MergeResultsError) as context:
            merged.write_merged_csv(columns, rows, blocker / 'merged.csv')
        self.assertEqual(context.exception.code, 'WRITE_FAILED')

    def test_run_merge_results_reports_the_destination_and_row_count(self):
        self.build_scope()
        output = self.temp_dir / 'out' / 'merged-results.csv'
        summary = merged.run_merge_results(self.scope, output=str(output))
        self.assertEqual(summary['output'], str(output))
        self.assertEqual(summary['rows'], 3)
        self.assertEqual(summary['iterations'], 3)
        self.assertEqual(len(self.read_csv(output)), 3)

    def test_cli_writes_the_file_and_prints_a_json_summary(self):
        self.build_scope()
        output = self.temp_dir / 'cli-merged.csv'
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = merged._main([
                'merge', '--root', str(self.scope),
                '--experiments', '1-100_1-100', '--output', str(output),
            ])
        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary['output'], str(output))
        self.assertEqual(summary['rows'], 2)
        self.assertEqual(summary['experiments'], ['1-100_1-100'])
        self.assertEqual(
            [row['DATE'] for row in self.read_csv(output)],
            ['2026-09-20_10-00-00', '2026-09-20_10-05-00'],
        )

    def test_cli_streams_the_csv_to_stdout_and_the_summary_to_stderr(self):
        self.build_scope()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = merged._main(['merge', '--root', str(self.scope), '--output', '-'])
        self.assertEqual(exit_code, 0)
        parsed = list(csv.DictReader(io.StringIO(stdout.getvalue())))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]['IDENTIFIER'], '1-100_1-100')
        summary = json.loads(stderr.getvalue())
        self.assertEqual(summary['output'], '<stdout>')
        self.assertEqual(summary['rows'], 3)

    def test_cli_reports_errors_as_json_and_exits_with_one(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = merged._main([
                'merge', '--root', str(self.temp_dir / 'missing'), '--output',
                str(self.temp_dir / 'merged.csv'),
            ])
        self.assertEqual(exit_code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload['code'], 'ROOT_NOT_FOUND')
        self.assertIn('does not exist', payload['error'])

    def test_cli_reports_no_results_with_its_own_code(self):
        self.scope.mkdir(parents=True, exist_ok=True)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = merged._main(['merge', '--root', str(self.scope)])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(stdout.getvalue())['code'], 'NO_RESULTS_FOUND')


if __name__ == '__main__':
    unittest.main()
