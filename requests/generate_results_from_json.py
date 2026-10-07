#!/usr/bin/env python3
"""Generate the derived ``results_from_json.csv`` of every iteration missing it.

The experiment pipeline persists the per-request ``output.csv`` of an iteration but
never keeps the ``results_from_json.csv`` the dashboard uploads, so a large results
archive has to be filled in on demand. ``convert_to_csv.py`` already knows how to
convert one iteration (``convert_one``) and how to batch one experiment folder
(``run_batch``), but neither walks the whole results tree, whose iterations live two
levels below the archive root::

    results/Experiment_<TYPE>_<ts>/<profile>/<iteration>/results.json

This script scans a results root recursively, finds every iteration folder that has a
``results.json``, and generates the ``results_from_json.csv`` each of them is missing.
Iterations that already have the CSV are skipped, so the pass is idempotent and can be
re-run after a new experiment without touching the existing files. One bad iteration is
reported and skipped instead of aborting the whole run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from convert_to_csv import convert_one


RESULTS_FILENAME = "results.json"
DERIVED_FILENAME = "results_from_json.csv"
TOKEN_FILENAME = "input_tokens.json"


def _load_env(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` .env file, ignoring comments and blank lines."""
    env: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip()
    except Exception:
        pass
    return env


def _default_results_root() -> str:
    """Resolve RESULTS_DIR from the environment, then the repo ``.env``, then ``results``."""
    root_dir = Path(__file__).resolve().parent.parent
    env = _load_env(root_dir / ".env")
    results_dir = os.environ.get("RESULTS_DIR") or env.get("RESULTS_DIR") or "results"
    if not os.path.isabs(results_dir):
        results_dir = str(root_dir / results_dir)
    return results_dir


def find_iteration_dirs(root: Path) -> list[Path]:
    """Return every folder below *root* that directly contains a ``results.json``.

    The results tree nests iterations two levels below the archive root, but a legacy
    flat layout is handled too because the folder is located by the file it holds. Hidden
    folders (and any path passing through one) are ignored.
    """
    iteration_dirs: set[Path] = set()
    for results_path in root.rglob(RESULTS_FILENAME):
        if not results_path.is_file():
            continue
        iteration_dir = results_path.parent
        relative_parts = iteration_dir.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        iteration_dirs.add(iteration_dir)
    return sorted(iteration_dirs)


def generate_missing(
    root: Path,
    include_text: bool,
    progress_every: int | None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Generate the missing ``results_from_json.csv`` below *root*.

    Returns ``(converted, skipped, failed)``; with *dry_run* the files are reported
    without being written. Iterations whose CSV already exists are skipped, and a
    single failing iteration is counted and reported without stopping the pass.
    """
    iteration_dirs = find_iteration_dirs(root)
    print(
        f"[generate_results_from_json] scanning {root}: "
        f"{len(iteration_dirs)} iteration folder(s) with {RESULTS_FILENAME}",
        flush=True,
    )

    converted = 0
    skipped = 0
    failed = 0
    for iteration_dir in iteration_dirs:
        input_json = iteration_dir / RESULTS_FILENAME
        output_csv = iteration_dir / DERIVED_FILENAME
        if output_csv.exists():
            skipped += 1
            print(
                f"[generate_results_from_json] {iteration_dir}: "
                f"skipped (already converted)",
                flush=True,
            )
            continue
        if dry_run:
            converted += 1
            print(
                f"[generate_results_from_json] {iteration_dir}: "
                f"would generate {DERIVED_FILENAME}",
                flush=True,
            )
            continue
        try:
            rows = convert_one(
                input_json,
                output_csv,
                iteration_dir / TOKEN_FILENAME,
                include_text,
                progress_every,
            )
        except Exception as error:  # noqa: BLE001 - one bad iteration must not stop the pass
            failed += 1
            print(
                f"Warning: {iteration_dir}: failed: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        converted += 1
        print(
            f"[generate_results_from_json] {iteration_dir}: converted {rows:,} rows",
            flush=True,
        )

    return converted, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the derived results_from_json.csv of every iteration under a "
        "results root that is missing it (existing files are left untouched)."
    )
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path(_default_results_root()),
        help="Root directory to scan recursively "
        "(default: RESULTS_DIR from the environment or the repository .env)",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Also write the response_text column (larger output).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Log progress and peak RSS every N records/requests (0 disables progress logs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the iterations that would be converted without writing them.",
    )
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    if not args.results_root.is_dir():
        parser.error(f"Results root does not exist or is not a directory: {args.results_root}")

    converted, skipped, failed = generate_missing(
        args.results_root,
        args.include_text,
        args.progress_every or None,
        args.dry_run,
    )
    suffix = " (dry run)" if args.dry_run else ""
    print(
        f"generate_results_from_json summary: converted={converted} skipped={skipped} "
        f"failed={failed}{suffix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
