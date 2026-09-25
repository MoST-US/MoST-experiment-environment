# 04 — Code style and conventions

## Baseline

- Python 3.11. Use only dependencies already listed in `requirements.txt` (pandas, numpy, scipy,
  statsmodels, PyYAML, ...); do not add a runtime dependency for a small helper.
- Do not reformat whole files. The `Makefile` runs `ruff format` / `black --check` on upstream code,
  but the MoST harness files are not black-formatted — keep diffs minimal and local.

## Per-layer style

`experiment_automation.py` and the harness:

- Single-quoted strings, `_leading_underscore` module-level helpers, `Step N` comments that mirror
  the documented pipeline, and `print()` diagnostics with a stable prefix (`Warning:`, `Error:`,
  `Controlled stop:`, `Early evaluation failure:`).
- Prefer returning tuples/dicts over raising; only genuinely fatal pipeline errors raise
  `RuntimeError` (via `run_command(..., fail_on_error=True)`).
- Parsing helpers must stay tolerant: skip malformed entries (`_parse_tokens_list`,
  `_parse_int_list`) and never let one bad line abort a multi-hour experiment.
- Keep the loop safety rails when editing the iteration loop: the `max_iterations` guard, the
  `ITERATION_HARD_LIMIT` checks and the `stop_after_persist` / `persist_return_value` pattern that
  guarantees the last iteration is persisted before returning.

`requests/*.py`:

- Type hints in the PEP 604 style already used (`str | None`, `dict[str, float]`, `list[Path]`),
  module docstrings, and an `if __name__ == "__main__": main()` entry point.
- Each script resolves `RESULTS_DIR`/`REQUESTS_*` on its own, chdirs as needed, and wraps best-effort
  work in `try/except Exception` + `print(f"Warning: ...")`. Preserve that: they are invoked
  standalone with `python -u <absolute path>`.
- Duplicate the ~12-line `_load_env` / `_read_env_value` helper instead of importing a shared
  module (CWD and paths differ per script, and the scripts must remain independently runnable).
- Only `evaluate.py` uses `sys.exit(0/1)` as a verdict; other scripts exit non-zero only for real
  errors, and their exit codes are not contractual.

General:

- Use `pathlib` and avoid POSIX-only semantics in Python: the code is written on Windows and run on
  Linux.
- For new subprocesses use `sys.executable` plus an argument list (the remaining `shell=True` string
  commands are legacy — do not extend that pattern, and fix them when you touch that code).
- Keep each file's existing language: some `evaluate.py` docstrings are Spanish, the rest is
  English. Do not mass-translate.
- No logging framework — the Slurm log is the interface, so keep stdout lines stable (see 02).
- When a value has units or an interval boundary, say it in the docstring/comment (`FILTER_BUFFER`
  is seconds, `DURATION` accepts `s/m/h/d`, token intervals are inclusive).

## Things that must not change silently

- `.env` is never written by code; configuration flows through `os.environ`.
- `results.csv` column names/order, and the stdout strings other scripts parse (see 02).
- TRUE/FALSE semantics and the stage machine (see 03).
- Iteration folder naming (`store_results._derive_directory_name`) and archive naming
  (`Experiment_<EXPERIMENT_TYPE>_<timestamp>`), which downstream consumers may rely on.
