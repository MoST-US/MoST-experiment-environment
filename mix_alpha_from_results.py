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
    python mix_alpha_from_results.py -r Experiment_MST_2026-09-20_10-00-00 \
        -p '(1-100:1-100,300-600:100-300),(1-100:1-100,300-600:100-300,600-1000:1000-1500)'

Workload mixes (`--profiles`):
- One bracketed group is one workload mix, i.e. one bracketed entry of WORKLOAD_MIXES and one full
  experiment, and the groups are calibrated in the order they are written:

      --profiles '(profile,profile),(profile,profile,profile)'

  Every mix is calibrated on its own: its profiles are looked up in the archive, their CU costs are
  derived from the MIT/MST values of that mix (`sigma = max MST of the mix / MST of the profile`, so
  the most capable profile of the mix is 1.0 CU) and its alphas are computed for its own profile
  list. Neither the number of mixes nor the profiles of the other mixes change a mix (the alpha
  formula is invariant to the CU scale, so a shared baseline would give the same request ratios).
- A value without brackets stays a single mix, exactly like before this format was added, so
  `--profiles 1-100:1-100,300-600:100-300` keeps working. `[...]` groups are accepted as well,
  because that is how the .env WORKLOAD_MIXES value delimits its mixes.
- `--cu-shares` is the alpha_CU vector of one mix, applied to every mix of the list in the profile
  order of that mix; a mix whose calibrated profile count differs from the vector raises an error.
  Without it every mix uses its own balanced split 1/k.
- Inside a group the profiles are bare labels (`inMin-inMax:outMin-outMax`; single values and
  one-sided ranges allowed, e.g. `32:64`). The `X1-X2:Y1:Y2` spelling of the output range is read as
  `X1-X2:Y1-Y2` with a warning, and an alpha left over from the .env `(profile,alpha)` form is
  ignored with a warning, because the alphas are computed here.

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
- `--cu-shares` defaults to the balanced CU split 1/k of each mix, so a balanced mix needs no extra
  argument. The predicted mix rates printed at the end use the same linear CU model and are an
  estimate for `REQ_MIN_START` (one value per mix, by index), not a measurement.

Standalone (stdlib only) and independent from .env: it can be run before the environment is
configured, exactly like mix_alpha_calculator.py.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# mix_alpha_calculator.py (alpha math + rendering + verification) and workload_mix.py (the
# WORKLOAD_MIXES grammar) live next to this file, so the repository root is made importable no
# matter where the script is started from.
_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from mix_alpha_calculator import (  # noqa: E402  (needs the sys.path fix above)
    VERIFY_TOLERANCE,
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
# One bracketed --profiles group is one workload mix: `(profile,profile),...` is the documented form
# and `[...]` is accepted too, because that is how WORKLOAD_MIXES delimits its mixes in .env.
MIX_GROUP_RE = re.compile(r'[\[\(]([^\[\]\(\)]*)[\]\)]')
# Reason reported for a profile the archive cannot calibrate because it has no such interval.
NO_CELL_REASON = 'the archive has no cell with this interval'


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


def _normalise_profile_label(text):
    """Canonicalise one profile written in --profiles; returns None when it is not a profile label.

    `workload_mix` stays the single source of truth for the profile grammar (`32:64` and
    `32-32:64-64` are the same profile, and anything the loadgen would reject is rejected here). The
    only extra tolerance is the `X1-X2:Y1:Y2` spelling of the output range: a valid profile never
    carries two ':', so that form is unambiguous and is read as `X1-X2:Y1-Y2`.
    """
    label = str(text or '').strip()
    if not label:
        return None
    if label.count(':') > 1:
        in_part, out_part = label.split(':', 1)
        canonical = f'{in_part.strip()}:{out_part.strip().replace(":", "-")}'
        profile = _label_ranges(canonical)
        if profile is not None:
            print(
                f'Warning: profile "{label}" uses the "Y1:Y2" output-range form; reading it as '
                f'{profile["label"]} (canonical form: inMin-inMax:outMin-outMax).'
            )
        return profile
    return _label_ranges(label)


def _looks_like_alpha(text) -> bool:
    """True when a chunk can only be an alpha: a profile label always carries one ':'."""
    return ':' not in str(text) and _parse_number(text) is not None


def split_mix_groups(profiles_arg) -> list:
    """Split --profiles into the raw text of its mixes (one group per mix, in order).

    `(p,p),(p,p,p)` is the documented form and `[p,p],[p,p,p]` (the delimiters of the .env
    WORKLOAD_MIXES value) is accepted too. Without any bracket the whole value is a single mix, which
    keeps `--profiles 1-100:1-100,300-600:100-300` valid. Text left outside the brackets (for
    example an unbalanced tail) is ignored with a `Warning:` instead of being silently dropped.
    """
    text = str(profiles_arg or '').strip()
    if not text:
        return []
    groups = [group.strip() for group in MIX_GROUP_RE.findall(text)]
    if not groups:
        return [text]
    leftover = MIX_GROUP_RE.sub(',', text).strip(' \t\r\n,;')
    if leftover:
        print(f'Warning: ignoring text outside the bracketed mixes: "{leftover}"')
    return groups


def parse_profile_mixes(profiles_arg):
    """Parse --profiles into its mixes, each one keeping its own profile order.

    Returns `(mixes, skipped)`: every mix is `{'index', 'raw', 'labels', 'skipped'}` (canonical
    labels, in the order they were written) and `skipped` collects the `{'label', 'reason'}` entries
    dropped while parsing, in the shape of the JSON payload (each mix also carries the entries that
    concern it). Malformed chunks, leftovers of the .env `(profile,alpha)` form and a profile
    repeated inside one mix are reported with a `Warning:` and never abort the calibration. A mix
    without a valid profile is kept here so the mix indices stay aligned with the WORKLOAD_MIXES
    order, and is dropped once calibrated.
    """
    mixes = []
    skipped = []
    for index, raw_group in enumerate(split_mix_groups(profiles_arg), start=1):
        labels = []
        mix_skipped = []
        for chunk in raw_group.replace(';', ',').split(','):
            text = chunk.strip()
            if not text:
                continue
            if _looks_like_alpha(text):
                entry = {
                    'label': text,
                    'reason': ('alphas are computed by this script; pass the profile without its '
                               'alpha'),
                }
            else:
                profile = _normalise_profile_label(text)
                if profile is None:
                    entry = {
                        'label': text,
                        'reason': 'not a valid inMin-inMax:outMin-outMax profile label',
                    }
                elif profile['label'] in labels:
                    entry = {
                        'label': profile['label'],
                        'reason': ('the profile is repeated inside the mix; its first occurrence is '
                                   'kept'),
                    }
                else:
                    labels.append(profile['label'])
                    continue
            mix_skipped.append(entry)
            skipped.append(entry)
            print(f'Warning: skipping profile "{entry["label"]}": {entry["reason"]}')
        if not labels:
            print(f'Warning: skipping workload mix without a valid profile: "{raw_group}"')
        mixes.append({
            'index': index,
            'raw': raw_group,
            'labels': labels,
            'skipped': mix_skipped,
        })
    return mixes, skipped


def _cell_for_label(label, cells):
    """Archive cell of one canonical profile label, or None when the archive has no such cell."""
    profile = _label_ranges(label)
    if profile is None:
        return None
    return cells.get((
        profile['in_min'],
        profile['in_max'],
        profile['out_min'],
        profile['out_max'],
    ))


def _add_skip(skipped, mixes, label, reason) -> None:
    """Record one dropped profile in the payload list and in the mixes that requested it."""
    entry = {'label': label, 'reason': reason}
    skipped.append(entry)
    for mix in mixes:
        if label in mix['labels']:
            mix['skipped'].append(entry)


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
            'pass one value per profile in mix order (the vector is applied to every mix).'
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


def _resolve_parent_dir(labels, alphas, mix_value):
    """Report the `mix_...` folder the automation will create, plus its rendered value when it differs.

    `workload_mix.parse_workload_mixes` renormalises the alphas it reads, so a mix whose 6-decimal
    alphas do not sum to exactly 1 (for example 0.930233,0.046512,0.023256) produces a folder whose
    last decimal differs from the one `render_parent_dir` computes. The parsed folder is the one
    requests/store_results.py really creates (experiment_automation.py reads `workload_mix
    ['parent_dir']`), so it is the value reported here; the rendered one is returned only when it
    differs, so the caller can warn about it (None when they agree).
    """
    rendered = render_parent_dir(labels, alphas)
    parsed = parse_workload_mixes(mix_value)
    if len(parsed) == 1 and [profile['label'] for profile in parsed[0]['profiles']] == list(labels):
        folder = parsed[0]['parent_dir']
        return folder, (rendered if rendered != folder else None)
    return rendered, None


def _build_calibrated_mix(spec, row_by_label, cu_shares_text):
    """Calibrate one parsed mix into its sigmas, alphas, folder name and REQ_MIN_START prediction.

    Returns None (with a `Warning:`) when none of its profiles could be calibrated, so a mix with an
    unusable roster never hides the mixes that are fine. Every mix is calibrated on its own profile
    list: sigma uses the most capable profile *of that mix* as the 1.0 CU reference and the alphas
    are computed over its own profiles. That keeps the mixes independent (the alpha formula is
    invariant to a common CU factor, so a shared baseline would give the same request ratios).
    """
    labels = [label for label in spec['labels'] if label in row_by_label]
    if not labels:
        print(
            f'Warning: skipping mix {spec["index"]} ("{spec["raw"]}"): none of its profiles could be '
            'calibrated.'
        )
        return None
    rows = [row_by_label[label] for label in labels]
    req_mins = [row['req_min'] for row in rows]
    sigmas = sigma_from_req_min(req_mins)
    cu_shares, balanced = _resolve_cu_shares(cu_shares_text, len(labels))
    alphas = compute_request_ratios(sigmas, cu_shares)
    achieved = achieved_cu_shares(alphas, sigmas)
    predicted = sum(share * req_min for share, req_min in zip(cu_shares, req_mins))
    mix_value = render_mix(labels, alphas)
    parent_dir, rendered_parent_dir = _resolve_parent_dir(labels, alphas, mix_value)
    return {
        'index': spec['index'],
        'raw': spec['raw'],
        'k': len(labels),
        'balanced_alpha_cu': balanced,
        # Per-profile view (same fields as the flat payload of a single mix); 'alpha' is the rendered
        # 6-decimal value written in .env, while the mix-level 'alphas' keeps the exact ratio.
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
        'sigmas': sigmas,
        'target_alpha_cu': cu_shares,
        'alphas': alphas,
        'achieved_alpha_cu': achieved,
        'skipped_profiles': list(spec['skipped']),
        'workload_mixes': mix_value,
        'parent_dir': parent_dir,
        # Folder rendered from the exact alphas; None when it is the one workload_mix will create.
        'rendered_parent_dir': rendered_parent_dir,
        'predicted_requests_per_minute': predicted,
        'suggested_req_min_start': int(round(predicted)),
    }


def _single_profile_warning(mix, total) -> str:
    """Message printed when a mix ended up with a single calibrated profile."""
    label = mix['profiles'][0]['label']
    if total > 1:
        return (
            f'Warning: only one profile ({label}) of mix {mix["index"]} was calibrated; a mix needs '
            'at least two, so this WORKLOAD_MIXES entry is a single-profile (non-additive) '
            'experiment.'
        )
    return (
        f'Warning: only one profile ({label}) was calibrated; a mix needs at least two, so this '
        'WORKLOAD_MIXES value is a single-profile (non-additive) experiment.'
    )


def _print_mix_block(mix, total) -> None:
    """Print one mix: its header (when there are several), its sigmas, its target and its table."""
    if total > 1:
        print(f'--- Mix {mix["index"]}/{total}: {mix["raw"]} ---')
    print('sigma (CU per request, most capable calibrated profile = 1): '
          + ', '.join(format_number(sigma) for sigma in mix['sigmas']))
    print()
    target = ('balanced alpha_CU = 1/k (default)' if mix['balanced_alpha_cu']
              else 'alpha_CU from --cu-shares')
    print(f'k = {mix["k"]} profile(s)   |   target = {target}')
    print()
    labels = [profile['label'] for profile in mix['profiles']]
    print_table(labels, mix['sigmas'], mix['target_alpha_cu'], mix['alphas'],
                mix['achieved_alpha_cu'])
    print_summary(labels, mix['alphas'], mix['target_alpha_cu'], mix['achieved_alpha_cu'])
    print()
    if mix['k'] < 2:
        print(_single_profile_warning(mix, total))


def _print_report(mixes, rows, metadata) -> None:
    """Print the metadata, the calibration table of every calibrated profile and one block per mix.

    The table is per profile (not per mix occurrence) because a profile shared by several mixes is
    calibrated once; the sigmas of each mix are printed in its own block.
    """
    if metadata:
        print(metadata)
    print()
    _print_calibration_table(rows)
    for row in rows:
        if row['note'] != FINISHED_ROW_NOTE:
            print(f'Warning: profile "{row["label"]}": {row["note"]}')
    print()
    for mix in mixes:
        _print_mix_block(mix, len(mixes))



def _warn_duplicate_mixes(mixes) -> None:
    """Warn when two mixes render to the same WORKLOAD_MIXES entry.

    `workload_mix.parse_workload_mixes` keeps a duplicated mix only once, so the automation would run
    that entry once even though the calibration list contains it twice.
    """
    first_index = {}
    for mix in mixes:
        previous = first_index.get(mix['workload_mixes'])
        if previous is None:
            first_index[mix['workload_mixes']] = mix['index']
            continue
        print(
            f'Warning: mix {mix["index"]} renders to the same WORKLOAD_MIXES entry as mix '
            f'{previous}; workload_mix.parse_workload_mixes runs a duplicated mix only once.'
        )


def _verify_mixes(combined_value, mixes) -> None:
    """Re-parse the combined WORKLOAD_MIXES value through workload_mix.

    `mix_alpha_calculator.verify_mix` only verifies a single mix, so the multi-mix case is checked
    here with the same idea: one parsed mix per rendered mix, the same profiles in the same order and
    the same alphas after the 6-decimal rendering plus the renormalisation done by the parser.
    """
    parsed_mixes = parse_workload_mixes(combined_value)
    if len(parsed_mixes) != len(mixes):
        print(
            f'Warning: verification failed, workload_mix parsed {len(parsed_mixes)} mix(es) instead '
            f'of {len(mixes)} (duplicated or malformed entries).'
        )
        return
    deviation = 0.0
    for parsed, mix in zip(parsed_mixes, mixes):
        labels = [profile['label'] for profile in mix['profiles']]
        want = [float(format_alpha(alpha)) for alpha in mix['alphas']]
        total = sum(want) or 1.0
        expected = [alpha / total for alpha in want]
        parsed_labels = [profile['label'] for profile in parsed['profiles']]
        if parsed_labels != labels or len(parsed['profiles']) != len(expected):
            print(
                f'Warning: verification failed, workload_mix read the profiles of mix '
                f'{mix["index"]} as {parsed_labels} instead of {labels}.'
            )
            return
        deviation = max(
            deviation,
            max(
                (abs(profile['alpha'] - alpha)
                 for profile, alpha in zip(parsed['profiles'], expected)),
                default=0.0,
            ),
        )
    if deviation > VERIFY_TOLERANCE:
        print(f'Warning: verification mismatch, max alpha deviation {deviation:.2e}.')
        return
    print(f'Verification: OK ({len(mixes)} mix(es), {sum(mix["k"] for mix in mixes)} profile(s), '
          f'max deviation {deviation:.2e})')
    print('Verification: canonical mixes  ' + ','.join(mix['workload_mixes'] for mix in mixes))
    print('Verification: parent folders  ' + ', '.join(mix['parent_dir'] for mix in mixes))


def _warn_rendered_parent_dirs(mixes) -> None:
    """Warn when the folder rendered from the exact alphas is not the folder the run will create."""
    for mix in mixes:
        if mix['rendered_parent_dir']:
            print(
                f'Warning: mix {mix["index"]} renders to {mix["rendered_parent_dir"]}, but '
                f'workload_mix creates {mix["parent_dir"]} (it renormalises the 6-decimal alphas); '
                f'using {mix["parent_dir"]}.'
            )


def _print_final_block(mixes, combined_value) -> None:
    """Print the values to paste in .env: WORKLOAD_MIXES, the parent folders and REQ_MIN_START."""
    _warn_duplicate_mixes(mixes)
    _warn_rendered_parent_dirs(mixes)
    if len(mixes) > 1:
        print(f'WORKLOAD_MIXES={combined_value}')
        print('parent folders: ' + ', '.join(mix['parent_dir'] for mix in mixes))
        print(
            'Predicted mix request rates per mix (linear CU model, sum alpha_CU * REQ_MIN): '
            + ', '.join(format_number(mix['predicted_requests_per_minute']) for mix in mixes)
            + ' requests/min'
        )
        print('Suggested REQ_MIN_START='
              + ','.join(str(mix['suggested_req_min_start']) for mix in mixes))
        _verify_mixes(combined_value, mixes)
        return
    mix = mixes[0]
    labels = [profile['label'] for profile in mix['profiles']]
    print(f'WORKLOAD_MIXES={combined_value}')
    print(f'parent folder: {mix["parent_dir"]}')
    print('Predicted mix request rate (linear CU model, sum alpha_CU * REQ_MIN): '
          f'{format_number(mix["predicted_requests_per_minute"])} requests/min')
    print(f'Suggested REQ_MIN_START={mix["suggested_req_min_start"]}')
    verify_mix(combined_value, labels, mix['alphas'])


def calibrate(results_name, profiles, cu_shares_text=None, results_dir_arg=None) -> dict:
    """Calibrate the requested workload mixes from an archive and print the whole report.

    `profiles` is either one mix (`p,p`) or several bracketed mixes (`(p,p),(p,p,p)`); every mix is
    calibrated on its own profile list, so the mixes of a list are independent of each other.

    Returns the calibration payload (also emitted as JSON by --json and used by the unit tests).
    Raises SystemExit with an `Error:` message when the archive as a whole is unusable or when a mix
    cannot honour --cu-shares; a single profile without a cell or without a usable REQ_MIN is skipped
    with a `Warning:` so the other profiles (and mixes) stay calibratable.
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

    mixes_spec, skipped = parse_profile_mixes(profiles)
    if not mixes_spec:
        raise SystemExit('Error: no profile to calibrate; pass --profiles <profile>,<profile>.')

    # The archive lookup and the calibration table are per profile: a profile used by several mixes
    # is calibrated once and every mix then reads its own sigmas from those values.
    labels_in_order = [label for mix in mixes_spec for label in mix['labels']]
    cell_by_label = {}
    missing = []
    for label in dict.fromkeys(labels_in_order):
        cell = _cell_for_label(label, cells)
        if cell is None:
            missing.append(label)
        else:
            cell_by_label[label] = cell
    for label in missing:
        print(f'Warning: skipping profile "{label}": {NO_CELL_REASON}')
        _add_skip(skipped, mixes_spec, label, NO_CELL_REASON)
    if missing:
        print(f'Warning: available cells in the archive: {_format_available_cells(cells)}')

    rows = []
    records = []
    for label in dict.fromkeys(labels_in_order):
        cell = cell_by_label.get(label)
        if cell is None:
            continue
        value, note, record = select_calibration_value(cell['records'])
        if value is None or record is None:
            print(f'Warning: skipping profile "{label}": {note}')
            _add_skip(skipped, mixes_spec, label, note)
            continue
        records.extend(cell['records'])
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

    row_by_label = {row['label']: row for row in rows}
    mixes = []
    for spec in mixes_spec:
        mix = _build_calibrated_mix(spec, row_by_label, cu_shares_text)
        if mix is not None:
            mixes.append(mix)
    if not mixes:
        raise SystemExit('Error: none of the requested mixes could be calibrated.')

    metadata = _metadata_summary(records)
    _print_report(mixes, rows, metadata)
    combined_mix_value = ','.join(mix['workload_mixes'] for mix in mixes)
    _print_final_block(mixes, combined_mix_value)

    payload = {
        'archive': str(archive),
        'results_dir': str(results_dir),
        'metadata': metadata,
        'calibrated_profiles': [
            {
                'label': row['label'],
                'cell': row['cell'],
                'source': row['source'],
                'iterations': row['iterations'],
                'req_min': row['req_min'],
                'stage': row['stage'],
                'termination_reason': row['termination_reason'],
                'value_note': row['note'],
            }
            for row in rows
        ],
        'skipped_profiles': skipped,
        'mixes': mixes,
        'workload_mixes': combined_mix_value,
        'parent_dirs': [mix['parent_dir'] for mix in mixes],
        'predicted_requests_per_minute_per_mix': [
            mix['predicted_requests_per_minute'] for mix in mixes
        ],
        'suggested_req_min_starts': [mix['suggested_req_min_start'] for mix in mixes],
    }
    if len(mixes) == 1:
        # Flat keys of a single-mix calibration: kept for the consumers of the pre-group format, so
        # `-p 1-100:1-100,300-600:100-300` produces the same JSON as before.
        only = mixes[0]
        payload.update({
            'k': only['k'],
            'balanced_alpha_cu': only['balanced_alpha_cu'],
            'profiles': only['profiles'],
            'parent_dir': only['parent_dir'],
            'predicted_requests_per_minute': only['predicted_requests_per_minute'],
            'suggested_req_min_start': only['suggested_req_min_start'],
        })
    return payload


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
        help='Profiles of one mix, in mix order, each inMin-inMax:outMin-outMax (single values and '
        'one-sided ranges allowed, e.g. 32:64); every profile is looked up in the archive. Several '
        'mixes: one bracketed group each, e.g. (1-100:1-100,300-600:100-300),'
        '(32:64,600-1000:1000-1500), calibrated independently',
    )
    parser.add_argument(
        '--cu-shares',
        type=str,
        default=None,
        help='Comma-separated target CU-load fraction (alpha_CU) per profile, in mix order; the same '
        'vector is applied to every mix and it defaults to the balanced split 1/k of each mix, '
        'normalised when it does not sum to 1',
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
