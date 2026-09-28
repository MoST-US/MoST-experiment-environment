"""Derive the calibrated alphas of an additive WORKLOAD_MIXES experiment from a results archive.

MoST additive experiments need one `alpha` per workload profile (the share of requests routed to
it). Those alphas should follow the *capacity* of the system for each profile, and that capacity is
exactly what a previous TOKENS_LIST (non-additive) execution measured: the MIT/MST value of every
token-interval cell, i.e. the `REQ_MIN` of the `FINISHED=TRUE` row written by
requests/store_results.py.

This script reads such an archive (a directory named `Experiment_<TYPE>_<timestamp>` inside
`RESULTS_DIR`, created by experiment_automation.py at the end of a run), takes the MIT/MST value of
every requested profile, turns each value into the CU cost of one request of that profile

    sigma(p) = max_q MST(q) / MST(p)

(so the most capable profile of the calibration is 1.0 CU, exactly like the `--sigmas 1.0,20.0`
example of mix_alpha_calculator.py) and then reuses mix_alpha_calculator.py to convert a target
CU-load split alpha_CU into the request ratios written in .env:

    alpha(p) = (alpha_CU(p) / sigma(p)) / sum_q (alpha_CU(q) / sigma(q))   prop. alpha_CU(p)*MST(p)

The script only prints: it never reads nor writes .env, never touches results/, and does not modify
experiment_automation.py, workload_mix.py, requests/*.py or the loadgen. The printed
`WORKLOAD_MIXES=` line is meant to be copy-pasted into .env before the additive run.

Usage examples:
    python mix_alpha_from_results.py --results Experiment_MST_2026-09-20_10-00-00 \
        --profiles 1-100:1-100,300-600:100-300
    python mix_alpha_from_results.py -r Experiment_MST_2026-09-20_10-00-00 \
        -p 1-100:1-100,300-600:100-300 --cu-shares 0.5,0.5
    python mix_alpha_from_results.py -r results/Experiment_MST_2026-09-20_10-00-00 \
        -p 32:64,600-1000:1000-1500 --json

Notes:
- Profiles are always passed explicitly (in mix order) and are matched against the archive through
  the MIN/MAX_INPUT/OUTPUT_TOKENS columns of results.csv, not through the folder names, so both
  `in_out` (two-value) and `inMin-inMax_outMin-outMax` cells are found. A profile without a cell is
  skipped with a `Warning:` that lists the available cells instead of aborting the calibration.
- The archive must come from single-interval (TOKENS_LIST) experiments of the same model and
  hardware as the additive run: additive (WORKLOAD_MIX) archives have no per-profile interval, so
  they are rejected with an explicit error.
- The MIT/MST value of a cell is the `REQ_MIN` of its `FINISHED=TRUE` row; a cell whose experiment
  stopped in the stage-1 iteration hard limit falls back to `LARGEST_TRUE` (with a warning), and a
  cell with no usable value is skipped.
- `--cu-shares` defaults to the balanced CU split 1/k, so a balanced mix needs no extra argument.
  The predicted mix rate printed at the end uses the same linear CU model and is an estimate for
  `REQ_MIN_START`, not a measurement.

Standalone (stdlib only) and independent from .env: it can be run before the environment is
configured, exactly like mix_alpha_calculator.py.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# mix_alpha_calculator.py (alpha math + rendering + verification) and workload_mix.py (the
# WORKLOAD_MIXES grammar) live next to this file, so the repository root is made importable no
# matter where the script is started from.
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from mix_alpha_calculator import (  # noqa: E402  (needs the sys.path fix above)
    achieved_cu_shares,
    compute_request_ratios,
    format_alpha,
    format_number,
    print_summary,
    print_table,
    render_mix,
    render_parent_dir,
    verify_mix,
)
from workload_mix import parse_workload_mixes  # noqa: E402

# Name of the per-iteration summary file written by requests/store_results.py.
RESULTS_CSV = 'results.csv'
# Values of the results.csv contract (see requests/store_results.py).
FLAG_TRUE = 'TRUE'
FINISHED_ROW_NOTE = 'FINISHED=TRUE row'
# Columns of results.csv holding the token interval of a non-additive cell.
INTERVAL_COLUMNS = (
    'MIN_INPUT_TOKENS',
    'MAX_INPUT_TOKENS',
    'MIN_OUTPUT_TOKENS',
    'MAX_OUTPUT_TOKENS',
)
# Glob metacharacters: an archive name containing one is used as a glob as-is.
_GLOB_CHARS = '*?['


def _read_env_value(env_path, key, default=''):
    """Read one KEY=VALUE from an .env file (tolerant reader, duplicated on purpose).

    The harness scripts stay independently runnable and are started from different working
    directories, so each one carries its own small env reader instead of importing a shared module.
    """
    try:
        path = Path(env_path)
        if not path.exists():
            return default
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith('#') or '=' not in text:
                    continue
                if text.startswith('export '):
                    text = text[len('export '):].strip()
                name, value = text.split('=', 1)
                if name.strip() == key:
                    return value.strip().strip('"').strip("'")
    except Exception as exc:  # pragma: no cover - an unreadable .env is a warning, not a failure
        print(f'Warning: could not read {env_path}: {exc}')
    return default


def resolve_results_dir(results_dir_arg=None) -> Path:
    """Resolve the results tree: --results-dir, then RESULTS_DIR, then .env, then 'results'.

    Relative paths are resolved against the repository root (where .env lives), not against the
    current working directory, so the script behaves the same from any folder.
    """
    if results_dir_arg:
        candidate = Path(str(results_dir_arg).strip())
        return candidate if candidate.is_absolute() else (_ROOT_DIR / candidate)

    raw = os.environ.get('RESULTS_DIR')
    if not raw:
        raw = _read_env_value(_ROOT_DIR / '.env', 'RESULTS_DIR', '')
    if not raw:
        raw = _read_env_value(Path('.env'), 'RESULTS_DIR', '')
    if not raw:
        raw = 'results'
    candidate = Path(str(raw).strip())
    return candidate if candidate.is_absolute() else (_ROOT_DIR / candidate)


def resolve_archive(name, results_dir: Path) -> Path:
    """Return the archive directory named by `name` (a path, a folder name, or a glob).

    Resolution order: an existing path -> <results_dir>/<name> -> glob (containment when the name
    has no glob character). When several archives match, the newest is used (archives are named with
    a trailing timestamp) and the ignored ones are listed, so a partial name is safe to type.
    """
    raw = str(name or '').strip()
    if not raw:
        raise SystemExit('Error: no results archive given; pass --results <archive>.')

    direct = Path(raw)
    if direct.is_dir():
        return direct
    if not direct.is_absolute():
        nested = results_dir / raw
        if nested.is_dir():
            return nested

    pattern = raw if any(char in raw for char in _GLOB_CHARS) else f'*{raw}*'
    try:
        hits = sorted(
            (path for path in results_dir.glob(pattern) if path.is_dir()),
            key=lambda path: path.name,
        )
    except Exception as exc:
        raise SystemExit(f'Error: invalid archive pattern "{raw}": {exc}')
    if not hits:
        available = sorted(path.name for path in results_dir.glob('Experiment*') if path.is_dir())
        listing = ', '.join(available) if available else '<none>'
        raise SystemExit(
            f'Error: no results archive matches "{raw}" in {results_dir}. Available: {listing}.'
        )
    if len(hits) > 1:
        ignored = ', '.join(path.name for path in hits[:-1])
        print(
            f'Warning: {len(hits)} archives match "{raw}"; using the newest ({hits[-1].name}) '
            f'and ignoring: {ignored}'
        )
    return hits[-1]


def _parse_number(value):
    """Parse one numeric results.csv cell; None when it is empty or not a number."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value):
    """Parse one integer results.csv cell (token interval); None when it is not whole."""
    number = _parse_number(value)
    if number is None:
        return None
    rounded = int(number)
    return rounded if abs(number - rounded) < 1e-9 else None


def _row_flag(row, key) -> str:
    """Upper-cased, stripped value of a TRUE/FALSE flag column ('' when missing)."""
    return str(row.get(key) or '').strip().upper()


def _iter_rows(results_path: Path):
    """Yield the rows of one results.csv (one row per iteration of that cell)."""
    try:
        with results_path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return
            for row in reader:
                yield row
    except Exception as exc:
        print(f'Warning: could not read {results_path}: {exc}')


def _cell_key(row):
    """Token interval (in_min, in_max, out_min, out_max) of a row, or None when unavailable.

    Additive rows leave the four MIN/MAX_INPUT/OUTPUT_TOKENS columns empty on purpose, and a row
    without them cannot be matched to a mix profile, so those rows are ignored.
    """
    values = tuple(_parse_int(row.get(column)) for column in INTERVAL_COLUMNS)
    if any(value is None for value in values):
        return None
    return values


def _cell_record(row, results_path: Path, archive: Path) -> dict:
    """Normalise one results.csv row into the calibration record of its cell.

    Everything comes from the row itself (REQ_MIN, EVALUATION, FINISHED, LARGEST_TRUE, ...) plus the
    cell folder the row was found in, so the caller never has to know the archive layout.
    """
    relative = results_path.relative_to(archive)
    # <archive>/<cell>/<iteration_ts>/results.csv -> the cell is the first path component; an
    # iteration folder passed directly as the archive is reported with the archive name itself.
    parent_dir = relative.parts[0] if len(relative.parts) > 1 else archive.name
    return {
        'parent_dir': str(parent_dir),
        'source': str(relative.parent),
        'req_min': _parse_number(row.get('REQ_MIN')),
        'evaluation': _row_flag(row, 'EVALUATION'),
        'finished': _row_flag(row, 'FINISHED'),
        'largest_true': _parse_number(row.get('LARGEST_TRUE')),
        'smallest_false': _parse_number(row.get('SMALLEST_FALSE')),
        'stage': str(row.get('STAGE') or '').strip(),
        'termination_reason': str(row.get('TERMINATION_REASON') or '').strip(),
        'experiment_type': str(row.get('EXPERIMENT_TYPE') or '').strip(),
        'model': str(row.get('MODEL_USED') or '').strip(),
        'gpu_count': str(row.get('GPU_COUNT') or '').strip(),
        'duration': str(row.get('DURATION') or '').strip(),
        'workload_mix': str(row.get('WORKLOAD_MIX') or '').strip(),
        # WORKLOAD_MIX is only written by additive runs, so it identifies them even when the
        # ADDITIVE column is missing (older rows).
        'additive': bool(_row_flag(row, 'ADDITIVE') == FLAG_TRUE
                         or str(row.get('WORKLOAD_MIX') or '').strip()),
    }


def scan_cells(archive: Path):
    """Collect the calibration records of every token-interval cell of an archive.

    Returns `(cells, additive_rows)`: `cells` maps the interval tuple to `{'key': ..., 'records':
    [...]}` (one record per iteration row found for that cell) and `additive_rows` counts the rows
    that belong to an additive (WORKLOAD_MIX) experiment and cannot calibrate a profile.
    """
    cells: dict = {}
    additive_rows = 0
    for results_path in sorted(archive.rglob(RESULTS_CSV)):
        for row in _iter_rows(results_path):
            record = _cell_record(row, results_path, archive)
            if record['additive']:
                additive_rows += 1
                continue
            key = _cell_key(row)
            if key is None:
                continue
            cells.setdefault(key, {'key': key, 'records': []})['records'].append(record)
    return cells, additive_rows


def select_calibration_value(records):
    """Pick the MIT/MST value of one cell out of the iteration rows collected for it.

    Precedence, following the results.csv contract (the first usable candidate wins):

    1. the `FINISHED=TRUE` row with `EVALUATION=TRUE`: its `REQ_MIN` is the answer of the experiment;
    2. a `FINISHED=TRUE` row carrying `LARGEST_TRUE` (a stage-1 hard-limit stop finishes the
       experiment as failed, and then the confirmed lower bound is the usable value);
    3. a `FINISHED=TRUE` row with a `REQ_MIN` and no verdict flag;
    4. the largest `LARGEST_TRUE` of the cell (the experiment was truncated before finishing);
    5. the largest `REQ_MIN` among the sustainable (`EVALUATION=TRUE`) iterations.

    Returns `(value, note, record)`, or `(None, reason, None)` when the cell has no usable value.
    `record` is the row the value came from, so the caller can report its STAGE / TERMINATION_REASON
    and folder without re-deriving them.
    """
    finished = [record for record in records if record['finished'] == FLAG_TRUE]
    for record in reversed(finished):
        if record['evaluation'] == FLAG_TRUE and record['req_min'] is not None:
            return record['req_min'], FINISHED_ROW_NOTE, record
    for record in reversed(finished):
        if record['largest_true'] is not None:
            return (
                record['largest_true'],
                'experiment finished as failed (stage-1 iteration hard limit); using LARGEST_TRUE '
                'as the sustainable value',
                record,
            )
    for record in reversed(finished):
        if record['req_min'] is not None:
            return record['req_min'], 'FINISHED=TRUE row without an EVALUATION flag', record
    confirmed = [
        record for record in records if record['largest_true'] is not None
    ]
    if confirmed:
        best = max(confirmed, key=lambda record: record['largest_true'])
        return best['largest_true'], 'no FINISHED=TRUE row; using the largest confirmed TRUE', best
    sustainable = [
        record for record in records
        if record['evaluation'] == FLAG_TRUE and record['req_min'] is not None
    ]
    if sustainable:
        best = max(sustainable, key=lambda record: record['req_min'])
        return (
            best['req_min'],
            'no FINISHED=TRUE row and no LARGEST_TRUE; using the highest sustainable REQ_MIN',
            best,
        )
    return None, ('no usable REQ_MIN (no FINISHED=TRUE row, no LARGEST_TRUE and no sustainable '
                  'iteration)'), None


def _label_ranges(label):
    """Normalise one profile label through workload_mix; returns its canonical dict or None.

    Reusing the grammar of the automation is what makes `32:64` and `32-32:64-64` the same profile,
    and it rejects anything the loadgen would not accept as a WORKLOAD_MIXES profile.
    """
    mixes = parse_workload_mixes(f'[({label},1)]')
    if not mixes or not mixes[0].get('profiles'):
        return None
    return mixes[0]['profiles'][0]


def _format_available_cells(cells) -> str:
    """Render the intervals found in the archive (inMin-inMax:outMin-outMax, sorted)."""
    return ', '.join(
        f'{key[0]}-{key[1]}:{key[2]}-{key[3]}' for key in sorted(cells)
    ) or '<none>'


def _resolve_profiles(profiles_arg, cells):
    """Match every requested profile label to a cell of the archive, keeping the mix order.

    Returns `(matched, skipped)`: the `(label, cell_info)` pairs in mix order, plus the
    `(label, reason)` pairs that could not be matched. Unparsable labels and labels without a cell
    are only reported with a warning (never a fatal error), because one unavailable profile must not
    stop the calibration of the others; the warnings are printed here, the skipped list is returned
    so the caller can expose it (JSON payload) without duplicating the messages.
    """
    matched = []
    skipped = []
    for raw_label in str(profiles_arg or '').split(','):
        label = raw_label.strip()
        if not label:
            continue
        profile = _label_ranges(label)
        if profile is None:
            skipped.append((label, 'not a valid inMin-inMax:outMin-outMax profile label'))
            continue
        cell = cells.get((
            profile['in_min'],
            profile['in_max'],
            profile['out_min'],
            profile['out_max'],
        ))
        if cell is None:
            skipped.append((profile['label'], 'the archive has no cell with this interval'))
            continue
        matched.append((profile['label'], cell))

    for label, reason in skipped:
        print(f'Warning: skipping profile "{label}": {reason}')
    if skipped:
        print(f'Warning: available cells in the archive: {_format_available_cells(cells)}')
    return matched, skipped


def sigma_from_req_min(req_mins):
    """Per-profile CU cost derived from the MIT/MST values of the archive.

    sigma(p) = max_q MST(q) / MST(p): the most capable profile is the reference (1.0 CU per
    request) and the others cost as many times more as they are slower. Keeping the reference on
    the fastest profile means every sigma is >= 1, which is what the documented examples of
    mix_alpha_calculator.py assume.
    """
    baseline = max(req_mins)
    return [baseline / value for value in req_mins]


def _parse_float_list(text, flag) -> list:
    """Parse a comma/semicolon-separated list of floats, warning on unparsable entries.

    Kept in step with mix_alpha_calculator._parse_float_list: the calibration script has to accept
    exactly the same `--cu-shares 0.5,0.5` syntax.
    """
    values = []
    for chunk in str(text).replace(';', ',').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            print(f'Warning: ignoring unparsable {flag} entry "{chunk}" (expected a number)')
    return values


def _resolve_cu_shares(text, k):
    """Target CU-load fractions from --cu-shares, or the balanced split 1/k.

    Returns the vector and whether the balanced default was used. The vector does not have to sum
    to 1: compute_request_ratios normalises both vectors, so only the ratios matter.
    """
    if text is None:
        return [1.0 / k] * k, True
    shares = _parse_float_list(text, '--cu-shares')
    if len(shares) != k:
        raise SystemExit(
            f'Error: {len(shares)} alpha_CU value(s) in --cu-shares for {k} calibrated profile(s); '
            'pass one value per profile in mix order.'
        )
    if any(share < 0 for share in shares):
        raise SystemExit('Error: every alpha_CU must be >= 0.')
    if sum(shares) <= 0:
        raise SystemExit('Error: the alpha_CU values must not all be 0.')
    return shares, False


def _metadata_summary(records) -> str:
    """Describe the configuration the calibration came from, warning when the cells disagree.

    Alphas are only meaningful for cells measured on the same model/hardware/duration, so a mixed
    archive is reported instead of silently averaged.
    """
    columns = (
        ('EXPERIMENT_TYPE', 'experiment_type'),
        ('MODEL_USED', 'model'),
        ('GPU_COUNT', 'gpu_count'),
        ('DURATION', 'duration'),
    )
    parts = []
    for column, key in columns:
        values = sorted({record[key] for record in records if record[key]})
        if not values:
            continue
        if len(values) > 1:
            print(
                f'Warning: the calibrated cells disagree on {column} ({", ".join(values)}); '
                'alphas are only meaningful for cells measured on the same configuration.'
            )
        parts.append(f'{column}={",".join(values)}')
    return '   '.join(parts)


def _print_calibration_table(rows) -> None:
    """Print one line per calibrated profile: where its value comes from and what it is."""
    header = (
        f'{"#":>2}  {"profile":<22} {"cell":<20} {"iters":>5} {"REQ_MIN":>9} {"EVAL":>5} '
        f'{"FINISHED":>8}  source'
    )
    print(header)
    print('-' * len(header))
    for index, row in enumerate(rows, start=1):
        req_min = row['req_min']
        print(
            f'{index:>2}  {row["label"]:<22} {row["cell"]:<20} {row["iterations"]:>5} '
            f'{format_number(req_min):>9} {row["evaluation"] or "-":>5} '
            f'{row["finished"] or "-":>8}  {row["source"]}'
        )


def calibrate(results_name, profiles, cu_shares_text=None, results_dir_arg=None) -> dict:
    """Calibrate the requested profiles from an archive and print the whole report.

    Returns the calibration payload (also emitted as JSON by --json and used by the unit tests).
    Raises SystemExit with an `Error:` message when the archive as a whole is unusable; a single
    profile without a cell or without a usable REQ_MIN is skipped with a `Warning:` so the other
    profiles stay calibratable.
    """
    results_dir = resolve_results_dir(results_dir_arg)
    archive = resolve_archive(results_name, results_dir)
    print(f'Results archive: {archive}')

    cells, additive_rows = scan_cells(archive)
    if not cells:
        if additive_rows:
            raise SystemExit(
                f'Error: {archive} only holds additive (WORKLOAD_MIX) rows; the alphas of a mix '
                'must be calibrated from single-interval (TOKENS_LIST) experiments.'
            )
        raise SystemExit(f'Error: no {RESULTS_CSV} with a token interval found under {archive}.')
    print(f'Calibration cells found: {len(cells)} ({_format_available_cells(cells)})')
    if additive_rows:
        print(f'Warning: ignored {additive_rows} additive (WORKLOAD_MIX) row(s) of the archive.')

    rows = []
    records = []
    matched, skipped = _resolve_profiles(profiles, cells)
    for label, cell in matched:
        records.extend(cell['records'])
        value, note, record = select_calibration_value(cell['records'])
        if value is None or record is None:
            skipped.append((label, note))
            print(f'Warning: skipping profile "{label}": {note}')
            continue
        rows.append({
            'label': label,
            'cell': record['parent_dir'],
            'source': record['source'],
            'iterations': len(cell['records']),
            'req_min': value,
            'evaluation': record['evaluation'],
            'finished': record['finished'],
            'stage': record['stage'],
            'termination_reason': record['termination_reason'],
            'note': note,
        })
    if not rows:
        raise SystemExit('Error: none of the requested profiles could be calibrated.')

    labels = [row['label'] for row in rows]
    req_mins = [row['req_min'] for row in rows]
    k = len(rows)
    sigmas = sigma_from_req_min(req_mins)
    cu_shares, balanced = _resolve_cu_shares(cu_shares_text, k)
    alphas = compute_request_ratios(sigmas, cu_shares)
    achieved = achieved_cu_shares(alphas, sigmas)
    mix_value = render_mix(labels, alphas)
    parent_dir = render_parent_dir(labels, alphas)
    predicted = sum(share * req_min for share, req_min in zip(cu_shares, req_mins))

    metadata = _metadata_summary(records)
    if metadata:
        print(metadata)
    print()
    _print_calibration_table(rows)
    for row in rows:
        if row['note'] != FINISHED_ROW_NOTE:
            print(f'Warning: profile "{row["label"]}": {row["note"]}')
    print()
    print('sigma (CU per request, most capable calibrated profile = 1): '
          + ', '.join(format_number(sigma) for sigma in sigmas))
    print()
    target = 'balanced alpha_CU = 1/k (default)' if balanced else 'alpha_CU from --cu-shares'
    print(f'k = {k} profile(s)   |   target = {target}')
    print()
    print_table(labels, sigmas, cu_shares, alphas, achieved)
    print_summary(labels, alphas, cu_shares, achieved)
    print()
    if k < 2:
        print(
            f'Warning: only one profile ({labels[0]}) was calibrated; a mix needs at least two, so '
            'this WORKLOAD_MIXES value is a single-profile (non-additive) experiment.'
        )
    print(f'WORKLOAD_MIXES={mix_value}')
    print(f'parent folder: {parent_dir}')
    print('Predicted mix request rate (linear CU model, sum alpha_CU * REQ_MIN): '
          f'{format_number(predicted)} requests/min')
    print(f'Suggested REQ_MIN_START={int(round(predicted))}')
    verify_mix(mix_value, labels, alphas)

    return {
        'archive': str(archive),
        'results_dir': str(results_dir),
        'metadata': metadata,
        'k': k,
        'balanced_alpha_cu': balanced,
        'profiles': [
            {
                'label': row['label'],
                'cell': row['cell'],
                'source': row['source'],
                'iterations': row['iterations'],
                'req_min': row['req_min'],
                'sigma': sigma,
                'alpha': float(format_alpha(alpha)),
                'alpha_cu': share,
                'achieved_alpha_cu': real,
                'stage': row['stage'],
                'termination_reason': row['termination_reason'],
                'value_note': row['note'],
            }
            for row, sigma, share, alpha, real in zip(rows, sigmas, cu_shares, alphas, achieved)
        ],
        'skipped_profiles': [
            {'label': label, 'reason': reason} for label, reason in skipped
        ],
        'workload_mixes': mix_value,
        'parent_dir': parent_dir,
        'predicted_requests_per_minute': predicted,
        'suggested_req_min_start': int(round(predicted)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Derive the calibrated alphas of a WORKLOAD_MIXES additive experiment from the MIT/MST '
            'values recorded in a previous results archive'
        )
    )
    parser.add_argument(
        '-r',
        '--results',
        required=True,
        help='Results archive: a folder name under RESULTS_DIR (Experiment_<TYPE>_<timestamp>), a '
        'partial name or glob matching one, or a path to it',
    )
    parser.add_argument(
        '-p',
        '--profiles',
        required=True,
        help='Comma-separated profiles of the mix, in mix order, each inMin-inMax:outMin-outMax '
        '(single values and one-sided ranges allowed, e.g. 32:64); every profile is looked up in '
        'the archive',
    )
    parser.add_argument(
        '--cu-shares',
        type=str,
        default=None,
        help='Comma-separated target CU-load fraction (alpha_CU) per profile, in mix order; '
        'defaults to the balanced split 1/k and is normalised when it does not sum to 1',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Override the results tree (defaults to RESULTS_DIR, then .env, then "results")',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Also print the calibration as JSON (machine readable, for scripts and dashboards)',
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = calibrate(args.results, args.profiles, args.cu_shares, args.results_dir)
    if args.json:
        print()
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
