"""Parsing and normalisation of the WORKLOAD_MIXES knob (MoST additive experiments).

`WORKLOAD_MIXES` is a comma-separated list of workload mixes; every mix is written between
brackets and made of `(profile, alpha)` pairs:

    WORKLOAD_MIXES=[(1-100:1-100,0.5),(300-600:100-300,0.5)],[(1-100:1-100,1.0)]

Each `[...]` entry is one experiment. Inside a mix, every request picks a profile with
probability `alpha` (alphas are normalised so they add up to 1 across the mix). A profile
is a token interval `inMin-inMax:outMin-outMax`; single values (`1-100:100`) are accepted
and normalised to their two-value form. Profiles that repeat the same interval in one mix
have their alphas summed.

Parsing is tolerant on purpose: a malformed mix or pair is skipped with a `Warning:` so one
bad line never aborts a multi-hour experiment. `;` is accepted as an alternative separator
between pairs.

The module is standalone (stdlib only): `experiment_automation.py` imports it, while the
`requests/*.py` scripts keep their own duplicated helpers and never import it.
"""

import json
import re

MIX_RE = re.compile(r'\[([^\[\]]*)\]')
PAIR_RE = re.compile(r'\(([^()]*)\)')
TOKEN_RANGE_RE = re.compile(r'^(\d+)(?:-(\d+))?$')
# Anything outside this set would be unsafe (or surprising) in a folder name on Linux/Windows.
UNSAFE_DIR_CHARS_RE = re.compile(r'[^0-9A-Za-z_.@+-]')
ALPHA_DECIMALS = 6


def _parse_token_range(text):
    """Parse 'min-max' (or a single value) into an inclusive (min, max) tuple.

    Returns None when the text is not a range of positive integers: the loadgen requires positive
    token bounds, so a 0 boundary is rejected here (with a warning) instead of crashing the run.
    """
    match = TOKEN_RANGE_RE.match(str(text).strip())
    if not match:
        return None
    low = int(match.group(1))
    high = int(match.group(2)) if match.group(2) is not None else low
    if low <= 0 or high <= 0:
        return None
    if low > high:
        low, high = high, low
    return low, high


def _format_alpha(alpha):
    """Render an alpha compactly: 0.5, 0.333333, 1 (no trailing zeros)."""
    text = f'{float(alpha):.{ALPHA_DECIMALS}f}'.rstrip('0').rstrip('.')
    return text or '0'


def _parse_pair(text):
    """Parse one '(profile, alpha)' pair into a profile dict.

    Returns None when the pair is malformed (missing interval, bad number, alpha <= 0).
    """
    parts = str(text).rsplit(',', 1)  # the alpha is the last field; intervals contain no comma
    if len(parts) != 2:
        return None
    interval_text = parts[0].strip()
    alpha_text = parts[1].strip()
    if ':' not in interval_text:
        return None
    in_part, out_part = interval_text.split(':', 1)
    in_range = _parse_token_range(in_part)
    out_range = _parse_token_range(out_part)
    if in_range is None or out_range is None:
        return None
    try:
        alpha = float(alpha_text)
    except ValueError:
        return None
    if not alpha > 0:
        return None
    in_min, in_max = in_range
    out_min, out_max = out_range
    return {
        'label': f'{in_min}-{in_max}:{out_min}-{out_max}',
        'in_min': in_min,
        'in_max': in_max,
        'out_min': out_min,
        'out_max': out_max,
        'alpha': alpha,
    }


def _build_mix(profiles):
    """Normalise the alphas of one mix and derive its canonical/parent_dir/envelope."""
    total = sum(profile['alpha'] for profile in profiles)
    for profile in profiles:
        profile['alpha'] = profile['alpha'] / total
    canonical = '[' + ','.join(
        f"({profile['label']},{_format_alpha(profile['alpha'])})" for profile in profiles
    ) + ']'
    # Folder name built from safe fields only (no ':', spaces, parentheses or slashes).
    parent_dir = 'mix_' + '+'.join(
        f"{profile['label'].replace(':', '_')}@{_format_alpha(profile['alpha'])}"
        for profile in profiles
    )
    return {
        'canonical': canonical,
        'parent_dir': UNSAFE_DIR_CHARS_RE.sub('-', parent_dir),
        'profiles': profiles,
        'envelope': {
            'in_min': min(profile['in_min'] for profile in profiles),
            'in_max': max(profile['in_max'] for profile in profiles),
            'out_min': min(profile['out_min'] for profile in profiles),
            'out_max': max(profile['out_max'] for profile in profiles),
        },
    }

def parse_workload_mixes(value):
    """Parse WORKLOAD_MIXES into a list of mixes (malformed entries are skipped).

    Duplicate mixes (same canonical string) are ignored with a warning: they would produce
    the same parent folder and silently merge two experiments into one result folder.
    """
    mixes = []
    if not value:
        return mixes
    text = str(value).replace(';', ',')
    seen_canonical = set()
    for raw_mix in MIX_RE.findall(text):
        by_label = {}
        profiles = []
        for raw_pair in PAIR_RE.findall(raw_mix):
            profile = _parse_pair(raw_pair)
            if profile is None:
                print(
                    'Warning: skipping malformed entry '
                    f'"({raw_pair.strip()})" in WORKLOAD_MIXES'
                )
                continue
            existing = by_label.get(profile['label'])
            if existing is not None:
                # The same profile twice in one mix: its alphas are additive.
                existing['alpha'] += profile['alpha']
                continue
            by_label[profile['label']] = profile
            profiles.append(profile)
        if not profiles:
            print(
                'Warning: skipping workload mix without valid "(profile, alpha)" pairs: '
                f'[{raw_mix.strip()}]'
            )
            continue
        mix = _build_mix(profiles)
        if mix['canonical'] in seen_canonical:
            print(f"Warning: duplicate workload mix ignored: {mix['canonical']}")
            continue
        seen_canonical.add(mix['canonical'])
        mixes.append(mix)
    return mixes


def mix_spec_payload(mix, filename_factory):
    """JSON payload exported as WORKLOAD_MIX_SPEC for the loadgen and store_results.py.

    filename_factory(in_min, in_max) resolves the requests file of a profile. The same input
    interval always maps to the same file, so profiles that differ only in their output
    interval share it.
    """
    profiles = []
    for profile in mix['profiles']:
        profiles.append({
            'label': profile['label'],
            'in_min': profile['in_min'],
            'in_max': profile['in_max'],
            'out_min': profile['out_min'],
            'out_max': profile['out_max'],
            'alpha': profile['alpha'],
            'filename': filename_factory(profile['in_min'], profile['in_max']),
        })
    return json.dumps(
        {'mix': mix['canonical'], 'profiles': profiles},
        separators=(',', ':'),
        ensure_ascii=True,
    )


def output_bounds_for_input(mixes, in_min, in_max, fallback=None):
    """Union of the output intervals of every profile using this input interval.

    Requests files are cached per input interval, so a file has to cover the widest output
    interval any configured profile may draw from; the served output length is still decided
    per request from the profile interval (see WORKLOAD_MIX_SPEC). Returns `fallback` (the
    input interval when not given) when no configured profile matches.
    """
    low = None
    high = None
    for mix in mixes or []:
        for profile in mix['profiles']:
            if profile['in_min'] == in_min and profile['in_max'] == in_max:
                low = profile['out_min'] if low is None else min(low, profile['out_min'])
                high = profile['out_max'] if high is None else max(high, profile['out_max'])
    if low is None or high is None:
        return fallback if fallback is not None else (in_min, in_max)
    return low, high
