"""Compute calibrated request ratios (alpha) for additive WORKLOAD_MIXES experiments.

MoST runs additive load tests by routing every request to one workload profile with probability
`alpha`. Setting those alphas from uncalibrated request counts (e.g. 50/50) mixes profiles whose
unit costs differ by orders of magnitude: a profile with sigma = 20 CU next to one with
sigma = 1 CU ends up exerting ~95% of the total system CU pressure at a 50/50 request split, so
any interference caused by the light profile stays statistically invisible.

To control the *system pressure* split instead, pick a target CU-load vector alpha_CU (the
fraction of the total CU load every profile must exert, with sum(alpha_CU) = 1) and convert it
into the request ratio vector alpha the loadgen needs:

    alpha(p) = (alpha_CU(p) / sigma(p)) / sum_q (alpha_CU(q) / sigma(q))

where sigma(p) is the CU cost of one request of profile p (the per-profile characterization
constant, e.g. 1.0 CU for a light profile and 20.0 CU for a heavy one).

This script is a calculator only: it prints the alpha vector plus a ready-to-paste
`WORKLOAD_MIXES=` line for .env. It never reads nor writes .env and it does not modify
experiment_automation.py, workload_mix.py, requests/*.py or the loadgen.

Usage examples:
    python mix_alpha_calculator.py -k 2 --sigmas 1.0,20.0 --cu-shares 0.5,0.5
    python mix_alpha_calculator.py -k 2 --sigmas 1.0,20.0          # alpha_CU = 1/k (balanced)
    python mix_alpha_calculator.py -k 2                            # prompts for sigma and alpha_CU
    python mix_alpha_calculator.py -k 2 --sigmas 1,20 --labels 1-100:1-100,300-600:100-300 --verify

Notes:
- `--cu-shares` defaults to the balanced pressure (equipartition centroid) point
  alpha_CU(p_i) = 1/k for every i, so a balanced CU split needs no extra arguments.
- Alphas are rendered with the same 6-decimal, trailing-zero-stripped style as
  workload_mix.ALPHA_DECIMALS, so the pasted string matches what the automation normalises,
  records in ADDITIVE_EXPECTED_PROPORTIONS and uses in the mix_... folder name.
- sigma has to be supplied by hand: this repository has no CU characterization model to read it
  from. The CU split reported here is the *target* per-request pressure of the mix; the
  physically achieved mix is still the one measured at runtime by ADDITIVE_TRUE_PROPORTIONS.
- Prompts are only used when stdin is a terminal; in an unattended run (piped input, Slurm log,
  redirected stdin) every value must come from the flags, otherwise the script fails instead of
  hanging on a prompt.

Standalone (stdlib only) and independent from .env: it can be run before the environment is
configured.
"""

import argparse
import sys
from pathlib import Path

# Must match workload_mix.ALPHA_DECIMALS: the rendered alphas are part of the canonical mix
# (ADDITIVE_EXPECTED_PROPORTIONS) and of the mix_... folder name.
ALPHA_DECIMALS = 6
# Informational threshold: below this request share a profile is still valid, but its pressure on
# the system will be too small to observe statistically.
NEGLIGIBLE_ALPHA = 0.01
# Tolerance used to decide whether the printed alphas survived the 6-decimal rendering.
VERIFY_TOLERANCE = 1e-9


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Convert a target CU-load split (alpha_CU) into the request ratios (alpha) to write '
            'in WORKLOAD_MIXES'
        )
    )
    parser.add_argument(
        '-k',
        '--profiles',
        type=int,
        default=None,
        help='Number of workload profiles of the mix (prompted when omitted)',
    )
    parser.add_argument(
        '--sigmas',
        type=str,
        default=None,
        help='Comma-separated CU cost (sigma) of one request per profile, in mix order '
        '(prompted per profile when omitted)',
    )
    parser.add_argument(
        '--cu-shares',
        type=str,
        default=None,
        help='Comma-separated target CU-load fraction (alpha_CU) per profile, in mix order; '
        'defaults to the balanced split 1/k and is normalised when it does not sum to 1',
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Optional comma-separated profile labels (inMin-inMax:outMin-outMax, in mix order); '
        'enables printing the ready-to-paste WORKLOAD_MIXES line',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Re-parse the generated line through workload_mix.parse_workload_mixes to prove the '
        'automation will accept it (requires --labels)',
    )
    return parser


def _format_alpha(alpha) -> str:
    """Render an alpha exactly like workload_mix._format_alpha (6 decimals, no trailing zeros)."""
    text = f'{float(alpha):.{ALPHA_DECIMALS}f}'.rstrip('0').rstrip('.')
    return text or '0'


def _format_number(value) -> str:
    """Render a sigma / alpha_CU compactly: 1, 20.5, 0.333333 (no trailing zeros)."""
    text = f'{float(value):.{ALPHA_DECIMALS}f}'.rstrip('0').rstrip('.')
    return text or '0'


def _parse_float_list(text, label) -> list[float]:
    """Parse a comma/semicolon-separated list of floats, warning on unparsable entries."""
    values: list[float] = []
    for chunk in str(text).replace(';', ',').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            print(f'Warning: ignoring unparsable {label} entry "{chunk}" (expected a number)')
    return values


def _prompt_text(prompt) -> str:
    """Read one answer from the terminal.

    Refuses to prompt when stdin is not a terminal (piped input, Slurm log, redirected stdin):
    the prompt would block forever instead of failing, which is exactly what happens when the
    arguments are simply forgotten in an unattended run.
    """
    if not sys.stdin.isatty():
        raise SystemExit(
            'Error: no terminal to prompt on (stdin is not interactive); pass the values as '
            'arguments, e.g. -k 2 --sigmas 1,20 --cu-shares 0.5,0.5.'
        )
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit(
            'Error: no interactive input available; pass the values as command-line arguments.'
        )


def _prompt_float(prompt, default=None) -> float:
    """Ask for a number, returning `default` on an empty answer (None means it is required)."""
    while True:
        raw = _prompt_text(prompt)
        if not raw:
            if default is not None:
                return float(default)
            print('Please enter a number.')
            continue
        try:
            return float(raw)
        except ValueError:
            print('Please enter a number.')


def _prompt_int(prompt) -> int:
    """Ask for a positive integer."""
    while True:
        raw = _prompt_text(prompt)
        try:
            value = int(raw)
        except ValueError:
            print('Please enter a whole number.')
            continue
        if value < 1:
            print('Please enter a value >= 1.')
            continue
        return value


def _resolve_profiles(k_argument) -> int:
    """Number of profiles: from -k/--profiles, or prompted."""
    if k_argument is None:
        return _prompt_int('Number of workload profiles (k): ')
    return int(k_argument)


def _resolve_sigmas(args, k) -> list[float]:
    """Per-profile CU cost (sigma): from --sigmas, or prompted one by one."""
    if args.sigmas is not None:
        return _parse_float_list(args.sigmas, '--sigmas')
    print(f'No --sigmas given: enter the CU cost (sigma) of each of the {k} profiles.')
    return [
        _prompt_float(f'  sigma of profile {index + 1} (CU per request) [1.0]: ', 1.0)
        for index in range(k)
    ]


def _resolve_cu_shares(args, k) -> tuple[list[float], bool]:
    """Target CU-load fractions: from --cu-shares, or the balanced split 1/k.

    Returns the vector and whether the balanced (equipartition centroid) default was used.
    """
    if args.cu_shares is None:
        return [1.0 / k] * k, True
    return _parse_float_list(args.cu_shares, '--cu-shares'), False


def _resolve_labels(args, k) -> list[str]:
    """Profile labels: parsed from --labels, else cosmetic `p1..pk` placeholders."""
    if args.labels is None:
        return [f'p{index + 1}' for index in range(k)]
    labels = [chunk.strip() for chunk in str(args.labels).split(',') if chunk.strip()]
    if len(labels) != k:
        print(
            f'Warning: --labels provided {len(labels)} label(s) for {k} profiles; using '
            'placeholder labels p1..pk instead.'
        )
        return [f'p{index + 1}' for index in range(k)]
    return labels


def _validate(sigmas, cu_shares, k) -> str | None:
    """Check both vectors are usable; returns an error message, or None when they are."""
    if len(sigmas) != k:
        return f'expected {k} sigma value(s), got {len(sigmas)}'
    if len(cu_shares) != k:
        return f'expected {k} alpha_CU value(s), got {len(cu_shares)}'
    if any(sigma <= 0 for sigma in sigmas):
        return 'every sigma must be > 0, because it is the CU cost of one request'
    if any(share < 0 for share in cu_shares):
        return 'every alpha_CU must be >= 0'
    if sum(cu_shares) <= 0:
        return 'the alpha_CU values must not all be 0'
    return None


def compute_request_ratios(sigmas, cu_shares) -> list[float]:
    """Return the request ratios alpha matching a target alpha_CU vector.

    alpha(p) = (alpha_CU(p) / sigma(p)) / sum_q (alpha_CU(q) / sigma(q)), renormalised so the
    vector sums to 1 (the loadgen normalises the alphas of a mix anyway; normalising here keeps
    the printed vector identical to what ADDITIVE_EXPECTED_PROPORTIONS reports).
    """
    weights = [share / sigma for share, sigma in zip(cu_shares, sigmas)]
    total = sum(weights)
    return [weight / total for weight in weights]


def achieved_cu_shares(alphas, sigmas) -> list[float]:
    """CU-load fractions produced by a request ratio vector (alpha * sigma, normalised)."""
    loads = [alpha * sigma for alpha, sigma in zip(alphas, sigmas)]
    total = sum(loads)
    return [load / total for load in loads]


def _render_mix(labels, alphas) -> str:
    """Canonical mix value of the WORKLOAD_MIXES grammar: [(label,alpha),(label,alpha)]."""
    return '[' + ','.join(
        f'({label},{_format_alpha(alpha)})' for label, alpha in zip(labels, alphas)
    ) + ']'


def _render_parent_dir(labels, alphas) -> str:
    """Parent folder name the automation derives from this mix (see workload_mix._build_mix)."""
    return 'mix_' + '+'.join(
        f"{label.replace(':', '_')}@{_format_alpha(alpha)}"
        for label, alpha in zip(labels, alphas)
    )


def _print_table(labels, sigmas, cu_shares, alphas, achieved) -> None:
    """Print the per-profile comparison of target CU load against the resulting request share."""
    header = (
        f'{"#":>2}  {"profile":<24} {"sigma":>9} {"alpha_CU":>10} {"alpha":>11} '
        f'{"achieved":>10}'
    )
    print(header)
    print('-' * len(header))
    for index, (label, sigma, share, alpha, real) in enumerate(
        zip(labels, sigmas, cu_shares, alphas, achieved), start=1
    ):
        print(
            f'{index:>2}  {label:<24} {_format_number(sigma):>9} {_format_number(share):>10} '
            f'{_format_number(alpha):>11} {_format_number(real):>10}'
        )


def _print_summary(labels, alphas, cu_shares, achieved) -> None:
    """Print the alpha vector (exact and as written in .env), the residual and the warnings."""
    print()
    print('alpha (exact)     : ' + ', '.join(repr(alpha) for alpha in alphas))
    print('alpha (in .env)   : ' + ','.join(_format_alpha(alpha) for alpha in alphas))
    residual = max(abs(real - share) for real, share in zip(achieved, cu_shares))
    print(f'max |achieved - target| alpha_CU: {residual:.2e}')
    print()
    for label, alpha in zip(labels, alphas):
        if float(_format_alpha(alpha)) <= 0:
            print(
                f'Warning: profile "{label}" rounds to alpha 0 at {ALPHA_DECIMALS} decimals; '
                'WORKLOAD_MIXES would discard it. Increase its alpha_CU or reduce the sigma gap.'
            )
        elif alpha < NEGLIGIBLE_ALPHA:
            print(
                f'Warning: profile "{label}" receives {_format_number(alpha * 100)}% of the '
                'requests; its pressure on the system may be too small to measure.'
            )


def _verify(mix_value, labels, alphas) -> None:
    """Re-parse the generated mix through workload_mix to prove the automation accepts it."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from workload_mix import parse_workload_mixes
    except Exception as exc:  # pragma: no cover - only when workload_mix.py is missing
        print(f'Warning: verification skipped, could not import workload_mix: {exc}')
        return
    mixes = parse_workload_mixes(mix_value)
    if len(mixes) != 1 or len(mixes[0]['profiles']) != len(labels):
        print(
            'Warning: verification failed, the generated mix did not parse back to one mix of '
            f'{len(labels)} profile(s).'
        )
        return
    # The alphas actually stored are the rendered ones, renormalised by the parser.
    rendered = [float(_format_alpha(alpha)) for alpha in alphas]
    total = sum(rendered)
    expected = [value / total for value in rendered]
    parsed = [profile['alpha'] for profile in mixes[0]['profiles']]
    deviation = max(abs(real - want) for real, want in zip(parsed, expected))
    if deviation > VERIFY_TOLERANCE:
        print(f'Warning: verification mismatch, max alpha deviation {deviation:.2e}.')
        return
    print(
        f'Verification: OK (workload_mix parsed {len(parsed)} profile(s), max deviation '
        f'{deviation:.2e})'
    )
    print(f"Verification: canonical mix  {mixes[0]['canonical']}")
    print(f"Verification: parent folder  {mixes[0]['parent_dir']}")



def main() -> None:
    args = _build_parser().parse_args()

    k = _resolve_profiles(args.profiles)
    if k < 1:
        raise SystemExit('Error: the number of profiles must be >= 1.')

    sigmas = _resolve_sigmas(args, k)
    cu_shares, equipartition = _resolve_cu_shares(args, k)

    problem = _validate(sigmas, cu_shares, k)
    if problem is not None:
        raise SystemExit(f'Error: {problem}.')

    labels = _resolve_labels(args, k)

    total_share = sum(cu_shares)
    if abs(total_share - 1.0) > 1e-9:
        print(
            f'alpha_CU summed to {_format_number(total_share)}; normalised to 1 before '
            'converting.'
        )
    cu_shares = [share / total_share for share in cu_shares]

    if k == 1:
        print('Warning: a single profile is not a mix; alpha is 1 by construction.')

    alphas = compute_request_ratios(sigmas, cu_shares)
    achieved = achieved_cu_shares(alphas, sigmas)

    print()
    target = 'equipartition alpha_CU = 1/k' if equipartition else 'alpha_CU from --cu-shares'
    print(f'k = {k} profile(s)   |   target = {target}')
    print()
    _print_table(labels, sigmas, cu_shares, alphas, achieved)
    _print_summary(labels, alphas, cu_shares, achieved)

    if args.labels is None:
        print(
            'Tip: pass --labels inMin-inMax:outMin-outMax,... to also print the WORKLOAD_MIXES '
            'line.'
        )
        return

    mix_value = _render_mix(labels, alphas)
    print()
    print(f'WORKLOAD_MIXES={mix_value}')
    print(f'parent folder: {_render_parent_dir(labels, alphas)}')
    if args.verify:
        _verify(mix_value, labels, alphas)


if __name__ == '__main__':
    main()
