#!/usr/bin/env python3
"""Remove results_from_json.csv files that do not match the current converter."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from convert_to_csv import CSV_COLUMNS


TARGET_FILENAME = "results_from_json.csv"


def has_expected_header(path: Path) -> bool:
    """Return whether a CSV has exactly the header produced by the converter."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
    except (OSError, UnicodeError, csv.Error):
        return False
    return header == list(CSV_COLUMNS)


def remove_invalid_files(results_root: Path, dry_run: bool = False) -> tuple[int, int]:
    """Remove invalid derived CSV files below *results_root*.

    Returns ``(checked, removed)``. Files that cannot be read are treated as
    invalid so stale or corrupted derived files do not remain cached.
    """
    checked = 0
    removed = 0
    for path in results_root.rglob(TARGET_FILENAME):
        if not path.is_file():
            continue
        checked += 1
        if has_expected_header(path):
            print(f"Keeping valid file: {path}")
            continue
        print(f"{'Would remove' if dry_run else 'Removing'} invalid file: {path}")
        if not dry_run:
            try:
                path.unlink()
            except OSError as error:
                raise OSError(f"Unable to remove {path}: {error}") from error
        removed += 1
    return checked, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove stale results_from_json.csv files with an outdated header."
    )
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Root directory to scan recursively (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report invalid files without deleting them",
    )
    args = parser.parse_args()

    if not args.results_root.is_dir():
        parser.error(f"Results root does not exist or is not a directory: {args.results_root}")

    checked, removed = remove_invalid_files(args.results_root, args.dry_run)
    action = "would be removed" if args.dry_run else "removed"
    print(f"Checked {checked} {TARGET_FILENAME} file(s); {removed} {action}.")


if __name__ == "__main__":
    main()
