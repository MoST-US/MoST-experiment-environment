"""Merge the per-iteration results.csv files of a results archive into a single CSV.

Every experiment folder of a results scope holds one sub-folder per iteration
(``<experiment>/<iteration>/results.csv``). The dashboard downloads those files individually
(as a ZIP that preserves the folder structure) or as this merged CSV. Merging keeps the
experiment identity in an ``IDENTIFIER`` column (for example ``1-100_1-100``) and the iteration
timestamp in a ``DATE`` column, so the rows stay attributable after the files are concatenated.

Shared by the MoST experiment environment and the MoST API (spawned via
``python MergeResultsCsv.py merge --root <results scope> --output <file>``).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

IDENTIFIER_COLUMN = "IDENTIFIER"
DATE_COLUMN = "DATE"
RESULTS_FILENAME = "results.csv"
# Additive (`WORKLOAD_MIXES`) folders are named `mix_...` and are not interval-matrix cells.
ADDITIVE_EXPERIMENT_PREFIX = "mix_"
# Mirrors DATE_KEYS in the dashboard (src/App.jsx): the first date-like column holding a value
# wins, otherwise the iteration folder name (`YYYY-MM-DD_HH-MM-SS`) is used.
DATE_COLUMN_CANDIDATES = ("Date", "date", "Timestamp", "timestamp", "created_at")


class MergeResultsError(Exception):
    """Raised when the merged CSV cannot be built."""

    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


def parse_experiment_list(raw_value):
    """Split a comma separated experiment list into clean, unique, order preserving names."""
    names = []
    for token in str(raw_value or "").split(","):
        name = token.strip()
        if name and name not in names:
            names.append(name)
    return names


def normalize_experiments(experiments):
    """Accept the CLI string form or an iterable and return a clean list of experiment names."""
    if experiments is None:
        return []
    if isinstance(experiments, (list, tuple, set)):
        return parse_experiment_list(",".join(str(item) for item in experiments))
    return parse_experiment_list(experiments)


def list_experiment_names(root, include_additive=False):
    """Experiment folders of a results scope, sorted by name.

    Additive (``mix_...``) folders are skipped unless ``include_additive`` is set.
    """
    try:
        names = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    except OSError as error:
        raise MergeResultsError(
            'Unable to read results root "%s": %s' % (root, error), "ROOT_NOT_FOUND"
        )

    if include_additive:
        return names
    return [name for name in names if not name.lower().startswith(ADDITIVE_EXPERIMENT_PREFIX)]


def list_iteration_csvs(experiment_dir):
    """``(iteration folder name, results.csv path)`` pairs of one experiment, ordered by name."""
    try:
        iteration_names = sorted(entry.name for entry in experiment_dir.iterdir() if entry.is_dir())
    except OSError:
        return []

    iteration_csvs = []
    for name in iteration_names:
        candidate = experiment_dir / name / RESULTS_FILENAME
        if candidate.is_file():
            iteration_csvs.append((name, candidate))
    return iteration_csvs


def read_results_rows(csv_path):
    """Read the header and the non-empty data rows of one results.csv."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = [str(name).strip() for name in next(reader)]
        except StopIteration:
            return [], []
        rows = [row for row in reader if any(str(value).strip() != "" for value in row)]
    return fieldnames, rows


def resolve_iteration_date(record, iteration_name):
    """Timestamp of an iteration: first date-like source column, else the iteration folder name."""
    for candidate in DATE_COLUMN_CANDIDATES:
        value = record.get(candidate)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return iteration_name


def merge_results_csv(
    root,
    experiments=None,
    include_additive=False,
    identifier_column=IDENTIFIER_COLUMN,
    date_column=DATE_COLUMN,
):
    """Merge every results.csv of a results scope.

    Returns ``(columns, rows, summary)``: ``columns`` is the merged header (the identifier and the
    date column first, then the union of the source columns in first-seen order, so archives whose
    results.csv gained columns over time still merge cleanly) and ``rows`` are dicts ready to be
    written. The identifier/date columns are always (re)built by this module, which keeps the
    experiment folder name and the iteration timestamp next to every merged row.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise MergeResultsError('Results root "%s" does not exist.' % root, "ROOT_NOT_FOUND")

    requested = normalize_experiments(experiments)
    selected = (
        requested
        if requested
        else list_experiment_names(root_path, include_additive=include_additive)
    )

    source_columns = []
    rows = []
    used_experiments = []
    skipped = []
    iteration_count = 0

    for experiment in selected:
        experiment_dir = root_path / experiment
        if not experiment_dir.is_dir():
            skipped.append("%s: experiment folder not found" % experiment)
            continue

        iteration_csvs = list_iteration_csvs(experiment_dir)
        if not iteration_csvs:
            skipped.append("%s: no iteration results.csv found" % experiment)
            continue

        rows_before = len(rows)
        for iteration_name, csv_path in iteration_csvs:
            try:
                fieldnames, csv_rows = read_results_rows(csv_path)
            except (OSError, csv.Error) as error:
                skipped.append("%s/%s: %s" % (experiment, iteration_name, error))
                continue

            for name in fieldnames:
                if name and name not in source_columns:
                    source_columns.append(name)

            for csv_row in csv_rows:
                record = {}
                for index, name in enumerate(fieldnames):
                    record[name] = csv_row[index] if index < len(csv_row) else ""
                record[identifier_column] = experiment
                record[date_column] = resolve_iteration_date(record, iteration_name)
                rows.append(record)

            iteration_count += 1

        if len(rows) > rows_before:
            used_experiments.append(experiment)

    if not rows:
        raise MergeResultsError("No results.csv files found in %s." % root, "NO_RESULTS_FOUND")

    columns = [identifier_column, date_column]
    for name in source_columns:
        if name not in columns:
            columns.append(name)

    # Rows written with an older/narrower header get '' instead of a missing key, so every merged
    # row exposes the full union of the source columns.
    for row in rows:
        for name in columns:
            row.setdefault(name, "")

    summary = {
        "rows": len(rows),
        "iterations": iteration_count,
        "experiments": used_experiments,
        "columns": columns,
        "skipped": skipped,
    }
    return columns, rows, summary


def _write_rows(handle, columns, rows):
    """Write the merged header and rows, keeping the multiline/quoted fields of results.csv."""
    writer = csv.DictWriter(
        handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)


def write_merged_csv(columns, rows, output=None):
    """Write the merged table to ``output``; ``-``/``None`` streams it to stdout.

    Returns the destination as a string (``<stdout>`` when the CSV went to stdout).
    """
    if output in (None, "", "-"):
        _write_rows(sys.stdout, columns, rows)
        return "<stdout>"

    destination = Path(output)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8", newline="") as handle:
            _write_rows(handle, columns, rows)
    except OSError as error:
        raise MergeResultsError(
            'Unable to write "%s": %s' % (destination, error), "WRITE_FAILED"
        )
    return str(destination)


def run_merge_results(
    root,
    experiments=None,
    include_additive=False,
    output=None,
    identifier_column=IDENTIFIER_COLUMN,
    date_column=DATE_COLUMN,
):
    """Build the merged CSV, write it and return the JSON serialisable summary."""
    columns, rows, summary = merge_results_csv(
        root,
        experiments=experiments,
        include_additive=include_additive,
        identifier_column=identifier_column,
        date_column=date_column,
    )
    summary["output"] = write_merged_csv(columns, rows, output)
    return summary


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="MergeResultsCsv",
        description="Merge the per-iteration results.csv files of a results scope into one CSV.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser(
        "merge", help="Merge every <experiment>/<iteration>/results.csv of a results scope."
    )
    merge_parser.add_argument(
        "--root", required=True, help="Results scope directory holding the experiment folders."
    )
    merge_parser.add_argument(
        "--experiments",
        default="",
        help="Comma separated experiment folders to merge (default: every folder of the scope).",
    )
    merge_parser.add_argument(
        "--include-additive",
        action="store_true",
        help="Also merge the additive (mix_...) experiment folders.",
    )
    merge_parser.add_argument(
        "--output",
        default="-",
        help="Destination CSV file, or '-' to stream the CSV to stdout (default: '-').",
    )
    merge_parser.add_argument(
        "--identifier-column",
        default=IDENTIFIER_COLUMN,
        help="Column holding the experiment name (default: %s)." % IDENTIFIER_COLUMN,
    )
    merge_parser.add_argument(
        "--date-column",
        default=DATE_COLUMN,
        help="Column holding the iteration timestamp (default: %s)." % DATE_COLUMN,
    )

    args = parser.parse_args(argv)

    try:
        summary = run_merge_results(
            root=args.root,
            experiments=args.experiments,
            include_additive=args.include_additive,
            output=args.output,
            identifier_column=args.identifier_column,
            date_column=args.date_column,
        )
    except MergeResultsError as error:
        print(json.dumps({"error": error.message, "code": error.code}))
        return 1
    except Exception as error:
        print(json.dumps({"error": str(error), "code": "UNKNOWN_ERROR"}))
        return 1

    # Streamed CSVs own stdout, so the JSON summary moves to stderr in that case.
    summary_stream = sys.stderr if args.output in (None, "", "-") else sys.stdout
    print(json.dumps(summary), file=summary_stream)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
