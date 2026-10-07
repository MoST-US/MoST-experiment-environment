---
paths:
  - "experiment_automation.py"
  - "generate_requests.py"
  - "requests/**"
  - "fmperf/**"
  - "examples/**"
---

# 05 — Validating changes in this project

## Reality check

The full pipeline cannot be validated locally: it needs a reachable inference endpoint
(vLLM/TGIS), a GPU, and `DURATION` per iteration (hours in total), and it is normally launched
through Slurm. Validate instead with syntax checks, the existing unit tests, and sandboxed runs of
individual scripts against synthetic fixtures. Never point a validation run at the real `results/`
directory.

## Preconditions

- A `.env` must exist before importing anything from `fmperf` (`fmperf/utils/constants.py` executes
  `load_dotenv()` and reads `os.environ["REQUESTS_FILENAME"]` at import time) and before running
  `experiment_automation.py`. Copy `.env.example` to `.env`, or export the variables you need.
- `pip install -r requirements.txt` for pandas/numpy/scipy/statsmodels/PyYAML.

## Steps

1. Syntax check without writing bytecode:
   `python -B -c "import ast,pathlib;ast.parse(pathlib.Path('experiment_automation.py').read_text(encoding='utf-8'))"`
2. Library import check: `python -c "import fmperf"` (requires `.env`).
3. Unit tests: `pytest fmperf/tests/` (equivalently `make test`).
   `fmperf/loadgen/test_collect_energy.py` covers the energy helpers via `unittest.mock`.
   `fmperf/tests/test_output_csv_contract.py` pins the `output.csv` column contract between
   `requests/convert_to_csv.py` and its consumers (`split_results.py`, `evaluate.py`,
   `store_results.py`); it runs standalone with `python fmperf/tests/test_output_csv_contract.py`
   when `pytest` is unavailable.
4. Sandbox the per-iteration scripts. Point `RESULTS_DIR` at a scratch directory outside the repo
   (for example `$env:TEMP\most_sandbox`) and place synthetic `output.csv`, `first_half.csv`,
   `second_half.csv` and `results.json` there, then run e.g.
   `python -B requests/store_results.py <model> <stage> <parent_dir> <in_range> <out_range> <req_min> <TRUE|FALSE> <median> "" "" "" "" "" "" FALSE`
   and assert the produced `results.csv` values, the created folder path (`<parent_dir>/<ts>`), the
   moved CSVs and the copied `results.json`.
5. For `evaluate.py` changes, prove the exit codes with synthetic halves: statistically identical
   halves → exit 0; a clearly degraded second half, or a `success_rate` below
   `SUCCESS_RATE_THRESHOLD` → exit 1 (check with `$LASTEXITCODE` on PowerShell, `$?` on bash).
6. For `analyze_metrics.py` changes, use a small synthetic `results.json` with token events and
   confirm the printed `Median responded requests per minute:` line is still parseable by the
   automation.
7. After touching any script, grep that the scraped stdout strings and argv indices still match the
   contracts in 02, for example:
   `Select-String -Path experiment_automation.py,requests\*.py -Pattern 'Median responded requests per minute|Median tokens per response'`
8. Linting is optional and currently blocked: `requirements-dev.txt` does not exist, so install
   `ruff`/`black`/`mypy` manually before `make format`, `make lint` or `make type-check`.

## Hygiene

- Remove sandbox artifacts after validating and keep the repository's `results/` free of test data.
- Never commit generated CSVs/JSON, `*.out` logs or `.env` (see 06).
- Report validation honestly: list the commands you ran and the parts that could not be exercised in
  this environment (anything endpoint-, GPU- or Slurm-dependent).
