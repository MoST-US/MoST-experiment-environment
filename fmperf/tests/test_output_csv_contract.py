"""Regression tests for the output.csv schema shared by the requests/ scripts.

convert_to_csv.py (the producer) was once rewritten to a new request-level column
schema while its consumers (requests/split_results.py, requests/evaluate.py and
requests/store_results.py) kept reading the legacy names ``received_timestamp`` /
``complete_response_time``. The break was silent because the automation downgraded the
failing pipeline steps to warnings, so every MST iteration was scored as a false
negative. This module pins the contract end to end:

    results.json  ->  convert_to_csv  ->  output.csv
                  ->  split_results   ->  first_half.csv / second_half.csv
                  ->  store_results   ->  iteration folder name

Run with:
    python fmperf/tests/test_output_csv_contract.py
or
    pytest fmperf/tests/test_output_csv_contract.py
"""

import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUESTS_DIR = REPO_ROOT / "requests"

# One request of the exporter's schema: request i is emitted at base + i * step, its
# tokens one second apart, so the request spans (tokens - 1) seconds of wall time.
BASE_EPOCH_SECONDS = 1_700_000_000  # 2023-11-14T22:13:20Z
NS_PER_SECOND = 1_000_000_000


def _token_record(worker, request, at_seconds, duration_ms=1000.0, ok=True, n_tokens=5):
    return {
        "worker_idx": worker,
        "request_idx": request,
        "timestamp": at_seconds * NS_PER_SECOND,
        "duration_ms": duration_ms,
        "ok": ok,
        "error": "" if ok else "boom",
        "n_tokens": n_tokens,
        "response": {"text": "ok"},
    }


class OutputCsvContractTest(unittest.TestCase):
    """Producer (convert_to_csv) and consumers (split/evaluate/store) agree on the schema."""

    @classmethod
    def setUpClass(cls):
        cls._sandbox = tempfile.TemporaryDirectory(prefix="most_output_csv_contract_")
        # Point RESULTS_DIR at the sandbox *before* importing split_results so its
        # module-level RESULTS_PATH never touches the repository results/ tree.
        os.environ["RESULTS_DIR"] = cls._sandbox.name
        for path in (REPO_ROOT, REQUESTS_DIR):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        cls.convert = importlib.import_module("convert_to_csv")
        cls.split = importlib.import_module("split_results")
        cls.store = importlib.import_module("store_results")
        try:
            cls.evaluate = importlib.import_module("evaluate")
        except Exception:  # heavy optional stack (statsmodels/matplotlib); not required
            cls.evaluate = None

    @classmethod
    def tearDownClass(cls):
        cls._sandbox.cleanup()

    def setUp(self):
        self.results_dir = Path(self._sandbox.name) / self._testMethodName
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Each consumer writes first_half.csv / second_half.csv into RESULTS_PATH; keep
        # that sandboxed and shrink the FILTER_BUFFER so a short fixture is enough.
        self._orig_results_path = self.split.RESULTS_PATH
        self._orig_buffer_seconds = self.split.BUFFER_SECONDS
        self.split.RESULTS_PATH = self.results_dir
        self.split.BUFFER_SECONDS = 1
        self.addCleanup(self._restore_split_module)

    def _restore_split_module(self):
        self.split.RESULTS_PATH = self._orig_results_path
        self.split.BUFFER_SECONDS = self._orig_buffer_seconds

    def _build_output_csv(self, requests=8, tokens=3, step_seconds=15):
        """Write results.json, then run it through the real convert_to_csv entry points."""
        records = []
        for request in range(requests):
            start = BASE_EPOCH_SECONDS + request * step_seconds
            for token in range(tokens):
                records.append(_token_record(0, request, start + token))
        results_json = self.results_dir / "results.json"
        results_json.write_text(json.dumps({"results": records}), encoding="utf-8")

        loaded = self.convert.load_records(results_json)
        input_tokens = self.convert.load_input_tokens(
            results_json.with_name("input_tokens.json")
        )
        rows = self.convert.aggregate_requests(loaded, input_tokens, False, None)
        output_csv = self.results_dir / "output.csv"
        self.convert.write_csv(rows, output_csv)
        return output_csv

    def test_pipeline_produces_two_halves_and_a_real_directory_name(self):
        output_csv = self._build_output_csv()

        first_half, second_half = self.split.process_experiment_data(output_csv)

        self.assertFalse(first_half.empty)
        self.assertFalse(second_half.empty)
        for half in (first_half, second_half):
            self.assertIn(self.split.TIMESTAMP_COLUMN, half.columns)
            self.assertIn(self.split.DURATION_COLUMN, half.columns)
        for name in ("first_half.csv", "second_half.csv"):
            self.assertTrue((self.results_dir / name).exists())

        # store_results must read the timestamp column by name; reading row[0]
        # (request_idx, an integer) is the bug that produced the folder name "0".
        cwd = os.getcwd()
        os.chdir(self.results_dir)
        try:
            dir_name, source = self.store._derive_directory_name()
        finally:
            os.chdir(cwd)
        self.assertEqual(source, "first_half.csv")
        self.assertRegex(dir_name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
        self.assertNotEqual(dir_name, "0")

    def test_producer_columns_match_consumer_expectations(self):
        columns = set(self.convert.CSV_COLUMNS)
        # split_results.py
        self.assertIn(self.split.TIMESTAMP_COLUMN, columns)
        self.assertIn(self.split.DURATION_COLUMN, columns)
        self.assertIn("successful_request", columns)
        self.assertIn("success_rate", columns)
        # evaluate.py
        if self.evaluate is not None:
            self.assertIn(self.evaluate.TIMESTAMP_COLUMN, columns)
            for metric in self.evaluate.METRICS:
                self.assertIn(metric, columns)

    def test_split_still_accepts_the_legacy_schema(self):
        """Backwards compatibility: old CSVs must keep splitting (no silent break)."""
        legacy_csv = self.results_dir / "legacy.csv"
        legacy_timestamps = [
            "20260107T120000",
            "20260107T120100",
            "20260107T120200",
            "20260107T120300",
            "20260107T120400",
            "20260107T120500",
        ]
        with legacy_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["received_timestamp", "complete_response_time", "success_rate", "success"]
            )
            for stamp in legacy_timestamps:
                writer.writerow([stamp, "1500.0", "99.0", "True"])

        first_half, second_half = self.split.process_experiment_data(legacy_csv)

        self.assertFalse(first_half.empty)
        self.assertFalse(second_half.empty)


if __name__ == "__main__":
    unittest.main()
