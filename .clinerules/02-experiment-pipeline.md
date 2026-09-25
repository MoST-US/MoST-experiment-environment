# 02 — Experiment pipeline and cross-file contracts

## Iteration pipeline (`experiment_automation.run_evaluation_pipeline`)

| # | Step | Command / effect | Working dir |
| --- | --- | --- | --- |
| 1 | load generation + measurement | `python -u -m fmperf.loadgen.run` (writes `RESULTS_DIR/results.json`; routes every request to a mix profile and tags it with `workload_profile` when `ADDITIVE=TRUE`) | repo root |
| 2 | `cd` | `os.makedirs(RESULTS_DIR)` + `os.chdir(RESULTS_DIR)` | `RESULTS_DIR` |
| 3 | convert | `python -u requests/convert_to_csv.py` → `output.csv` (per request: completion time, success, success_rate) | `RESULTS_DIR` |
| 4 | early metrics gate | `python -u requests/analyze_metrics.py .`; the automation scrapes `Median responded requests per minute:` and, for non-MIT runs, fails the iteration when it is `< 0.95 * REQ_MIN` | `RESULTS_DIR` |
| 5 | split | `python -u requests/split_results.py` → trims `FILTER_BUFFER` seconds from both ends and writes `first_half.csv` / `second_half.csv` | `RESULTS_DIR` |
| 6 | evaluate | `python -u requests/evaluate.py` (MST only; skipped for MIT). Exit code 0 = TRUE, non-zero = FALSE | `RESULTS_DIR` |
| 7 | persist | `python -u requests/store_results.py <args...>` → creates `parent_dir/<ts>/`, writes `results.csv` + `prompts.csv`, moves `output.csv`/`first_half.csv`/`second_half.csv`, copies `results.json` | `RESULTS_DIR` |

Rules:

- Keep this order. `evaluate.py` writes the `no_statistical_difference` column into every non-half
  CSV in `RESULTS_DIR`, and `store_results.py` then moves those CSVs into the iteration folder;
  running `evaluate.py` after `store_results.py` would lose that column.
- `requests/*.py` are **standalone subprocesses**: each resolves `RESULTS_DIR` itself (env → `.env`
  → `results`), chdirs into it, and duplicates its small env reader. Do not introduce shared imports
  between them and do not assume the caller's cwd.
- Workload files are model-agnostic: `fmperf/loadgen/run.py` discovers the model from
  `URL/v1/models` at runtime. `generate_requests.py` generates one workload per input-token interval
  (`REQUESTS_DIR/sample_requests_<in_min>-<in_max>.json`) and reuses it when it already exists.
- Only step 1 is fatal: `run_command(..., fail_on_error=True)` on loadgen raises, the token
  experiment is aborted (`{"aborted": True, "reason": ...}`) and the remaining experiments are
  skipped. Steps 3-6 only warn, and step 7 runs through `subprocess.run` without a return-code check
  (its own `try/except` prints `An error occurred: ...`).

## Exit-code contract

`requests/evaluate.py` communicates the verdict exclusively through `sys.exit(0)` (TRUE) /
`sys.exit(1)` (FALSE); `run_evaluation_pipeline` maps `returncode == 0` to `evaluation_success`.
Never reuse exit code `1` for a different meaning in that script.

## stdout is an API

- `requests/analyze_metrics.py` prints `Median responded requests per minute: <number>`, and
  `run_evaluation_pipeline` parses that exact prefix — rewording it silently disables the early gate.
- `experiment_automation.py` prints `Median tokens per response: <number>`; when the CLI value is
  missing, `requests/store_results.py::_extract_median_tokens_from_log` recovers it from the Slurm
  log with `Median tokens per response:\s*([0-9]+(?:\.[0-9]+)?)`. Keep the prefix and a plain,
  unformatted number.
- `requests/store_results.py` prints `Additive expected proportions: <json>` and
  `Additive true proportions: <json>` for additive rows (`{}` when unknown); they mirror the
  `ADDITIVE_*_PROPORTIONS` columns so the achieved mix is visible in the Slurm log.
- `requests/evaluate.py` also prints the `no_statistical_difference_overall` summary used when
  reading logs by hand; keep those lines meaningful.

## Environment handling (`set_process_env_for_run` and precedence)

- The automation **never writes `.env`**; it sets in-process env for the child processes (`REQ_MIN`,
  `MIN/MAX_INPUT_TOKENS`, `MIN/MAX_OUTPUT_TOKENS`, `REQUESTS_FILENAME`, `SERVICE_TYPE`, ...). Keep
  that invariant — it is what makes iterations reproducible and `.env` stable.
- Precedence is `os.environ` → `CONFIG` (`.env`, parsed once at import) → literal defaults in the
  `_get_*` helpers. Because `fmperf/utils/constants.py` calls `load_dotenv()` at import, `.env`
  values also become visible through `os.environ` in child processes.
- `requests/store_results.py` reads configuration only from `os.environ` plus argv, so anything it
  needs must already be exported (see the EXPERIMENT_TYPE note below).
- `set_process_env_for_run` derives `REQUESTS_FILENAME` by appending the input interval
  (`sample_requests_<in_min>-<in_max>.json`) and caches the unsuffixed base in
  `REQUESTS_FILENAME_BASE`, so repeated calls do not stack suffixes.
- Who reads what: `experiment_automation.load_env_config()` handles `TOKENS_LIST`, `WORKLOAD_MIXES`,
  `REQ_MIN_START`,
  `REQ_MIN_INCREASE_MULTIPLIER`, `STOP_THRESHOLD`, `EXPERIMENT_TYPE`, `DURATION`,
  `ITERATION_COOLDOWN_SECONDS`, `ITERATION_HARD_LIMIT`, `SERVICE_TYPE`, `USE_CASES_YAML`;
  `fmperf/loadgen/run.py` reads `TARGET`, `URL`, `MODEL_DISCOVERY_TIMEOUT`, `REQ_MIN`, `DURATION`,
  `BACKOFF`, `GRACE_PERIOD`, `REQUEST_TIMEOUT`, `TTFT_TIMEOUT`, `TPOT_TIMEOUT`,
  `WORKER_RPM_CAPACITY`, `MAX_WORKERS`, plus `ADDITIVE` and `WORKLOAD_MIX_SPEC` on additive runs;
  `fmperf/loadgen/generate-input.py` reads `PROMPTS_FILE`
  (only there — the automation hardcodes `oasst_roots_en_max1000_tokens.jsonl`),
  `FRAC_GREEDY`, `SAMPLE_SIZE`, `TARGET`, `URL`, `MODEL`; `requests/split_results.py` reads
  `FILTER_BUFFER`; `SUCCESS_RATE_THRESHOLD` is read by the automation and by `evaluate.py`.
  `THRESHOLD_TYPE`, `SUCCESS_RATE`, `RESULTS_ALL_FILENAME`, `CODE`, `NUM_USERS`, `SWEEP_USERS` are
  documented or used only by upstream fmperf paths, not by this pipeline.
- `requests/store_results.py` reads everything from `os.environ` plus argv, so the additive markers
  (`ADDITIVE`, `WORKLOAD_MIX`, `WORKLOAD_MIX_SPEC`) must already be exported when it is spawned; they
  travel in the environment on purpose (see the argv contract below).
- `EXPERIMENT_TYPE` must be passed to `store_results.py` through the child environment: the
  automation exports the resolved value (`os.environ['EXPERIMENT_TYPE'] = get_experiment_type()`)
  before spawning it, because `store_results.py` reads `os.environ` and never argv.

## Positional CLI contract: `experiment_automation` → `store_results.py`

Indices below are `sys.argv[1:]` inside `requests/store_results.py`, compact format:

| # | argument | value |
| --- | --- | --- |
| 0 | model | model used for the iteration (`MODEL`) |
| 1 | stage | `1` or `2` |
| 2 | parent_dir | `<in>-<in>_<out>-<out>`, or `mix_...` for an additive `WORKLOAD_MIXES` run — this is the format-detection key |
| 3 | in_range | `MIN-MAX` input tokens (on additive runs: the mix envelope, kept only for the layout detection; the persisted column is blank) |
| 4 | out_range | `MIN-MAX` output tokens (on additive runs: the mix envelope, ditto) |
| 5 | req_min | REQ_MIN of the iteration (largest TRUE on a stage-2 hard-limit stop) |
| 6 | evaluation | `TRUE` / `FALSE` |
| 7 | median | median response tokens (may be empty) |
| 8 | resolved_model | `MODEL_USED_RESOLVED` (may be empty) |
| 9 | termination_reason | empty unless a hard limit was hit |
| 10 | binary_distance_abs | empty unless a stage-2 hard limit |
| 11 | binary_distance_rel | empty unless a stage-2 hard limit |
| 12 | largest_true | confirmed largest TRUE (may be empty) |
| 13 | smallest_false | confirmed smallest FALSE (may be empty) |
| 14 | finished | `TRUE` / `FALSE` for the last iteration of the token experiment |

`store_results.py` selects this parser with a positive signature (`_compact_layout_signature`: second
argument is the stage `1`/`2`, third is a non-empty `parent_dir`, fourth/fifth are token intervals —
relaxed when `SERVICE_TYPE=SaaS`, where those slots carry the use-case id — and seventh is
`TRUE`/`FALSE`), and otherwise falls back to the legacy order `model, gpus, cpus, node, stage,
parent_dir, in_range, out_range, req_min, evaluation, [median, ...]`. The old `'_' in args[2]`
heuristic is gone: it silently switched parsers whenever a leading argument was added, and inserting
any argument in the middle still corrupts the run (wrong tokens/REQ_MIN/EVALUATION columns, iteration
folders nested under the wrong parent, and an `Experiment_*` archive folder that ends up empty
because `_archive_execution_results()` cannot find `results/<parent_dir>`).

Every invocation prints the layout that was selected (`CLI format: compact ...`, `CLI format:
legacy ...` or `CLI format: environment/no-args ...`). Because offset 3 of the compact layout is
indistinguishable from the legacy layout, a compact layout found at offset 1 or 2 is **not** parsed
as compact; instead it prints `Warning: compact CLI layout detected at argument index N instead of 0;
every results.csv column is shifted by N position(s)`. `run_experiment_for_tokens` prints a matching
`Warning: store_results.py argv contract mismatch` before spawning the child, so a misaligned run is
visible in the Slurm log instead of being silently mis-stored.

Fragilities to respect:

- Append new arguments at the **end** and update `store_results.py` in the same commit; never insert
  in the middle.
- `store_results.py` parses the legacy order positionally behind `len(args) >= N` guards, so extra
  arguments shift those guards as well.
- Keep `EXPERIMENT_TYPE` out of argv (see above): it travels through the environment, and
  `store_results.py` falls back to the repository-root `.env` (then `../.env`) when the variable is
  missing, so the first `results.csv` column is populated for manual/API invocations too.
- SaaS mode: `parent_dir` is the use-case id, which may contain no `_`, `/` or `\`. The signature
  relaxes the token-interval positions when `SERVICE_TYPE=SaaS`, so these runs are parsed as compact
  again; never rely on heuristics for new modes, and prefer explicit named arguments when refactoring.
- When refactoring, keep the legacy branch working or delete it deliberately: the MoST API/dashboard
  and older Slurm logs may still produce the legacy order.

## `results.csv` contract

- Columns are consumed by name downstream (`fmperf/utils/GpuCount.py::read_gpu_count_from_results_csv`)
  and by the MoST API/dashboard. Treat them as append-only: never rename or reorder; add new columns
  at the end of both the header and the row in `store_results.main()`.
- Keep them in sync across `store_results.py` (header + row) → `README.md` ("Relevant
  `results.csv` fields") → these rules.
- Empty means "unknown": the automation passes `''` for unconfirmed bounds and non-terminal
  iterations. Do not substitute `0`.
- One row per iteration; the final row of a token experiment is the one with `FINISHED=TRUE` and
  carries any `TERMINATION_REASON` / `BINARY_SEARCH_*` / `LARGEST_TRUE` / `SMALLEST_FALSE`
  information. `REQ_MIN` in that row is the answer (largest TRUE when a hard limit stopped stage 2).
- Additive (`WORKLOAD_MIXES`) rows: `WORKLOAD_MIX` carries the canonical mix and the
  `ADDITIVE_EXPECTED_PROPORTIONS` / `ADDITIVE_TRUE_PROPORTIONS` columns carry the normalised alphas
  and the observed share of each profile (all requests, failures included). The four
  `MIN/MAX_INPUT/OUTPUT_TOKENS` columns stay empty by design: the mix envelope is passed in argv
  only so `_compact_layout_signature` keeps detecting the compact layout, and the `.env` backfill is
  skipped for those columns. `REQ_MIN` / `EVALUATION` / `STAGE` / `FINISHED` keep their usual
  meaning, and `RESPONSES_WITHIN/OUTSIDE_EXPECTED_INTERVAL` are computed per profile and only over
  requests that produced tokens (a failed request has no output length to classify).
- Additive per-profile data travels in `WORKLOAD_MIX_SPEC` (environment, never argv). A missing or
  unusable spec degrades to empty additive columns with a `Warning:` instead of failing the store
  step, and `requests/store_results.py` resolves the per-profile requests files (prompts.csv, median
  prompt tokens, input tokens) through that spec.
