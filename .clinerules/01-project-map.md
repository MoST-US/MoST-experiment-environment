# 01 — Project map: what this repository is

MoST experiment environment = a vendored copy of the **fmperf** benchmark library **plus** a
MoST-specific experiment harness. Decide which layer you are touching before editing.

## Two layers

| Layer | Paths | Nature |
| --- | --- | --- |
| MoST harness (this project's own code) | `experiment_automation.py`, `requests/*.py`, `generate_requests.py`, `*.slurm`, `.env.example`, `experiment_setting.txt`, `README.md` | Actively developed: drives the experiments and writes the published results |
| Upstream fmperf (vendored) | `fmperf/**`, `examples/**`, `docs/**`, `Dockerfile`, `Makefile`, `setup.py`, `requirements.txt` | Upstream benchmark / k8s / energy code; keep changes minimal (see 07) |

Dead code: the root-level `loadgen` file is a stale copy of `fmperf/loadgen/run.py` and nothing
imports it. Never edit it (it is also excluded via `.clineignore`).

## Runtime shape

- Edited on Windows, **run on Linux + Slurm** inside conda env `fmperf-env`
  (`experiment_automation.slurm` runs `python -u experiment_automation.py`).
- Python 3.11. Dependencies come from `requirements.txt` only (pandas, numpy, scipy, statsmodels,
  PyYAML, ...). There is no `requirements-dev.txt`, although `Dockerfile` and `Makefile` reference
  one — `make install-dev/venv-dev/format/lint/type-check` will fail until it exists.
- Entry point: `python experiment_automation.py`, or `sbatch experiment_automation.slurm`.
- `fmperf/utils/constants.py` calls `load_dotenv()` **and** reads `os.environ["REQUESTS_FILENAME"]`
  at import time, so nothing runs (or imports cleanly) without a `.env` next to the code.

## Configuration

- `.env` is the single source of truth at run time and `.env.example` is its tracked
  documentation — update the example whenever a knob changes. `.env` must never be written by code
  (see 02 for the in-process env mechanism).
- `EXPERIMENT_TYPE` selects the evaluation method:
  - **MIT** (Maximum Instantaneous Throughput): short iterations (~2 min); find the requests/min
    point where throughput/response time plateaus.
  - **MST** (Maximum Sustainable Throughput): long iterations (~30 min); find the largest
    requests/min that keeps QoS (success rate, response-time stability between halves).
  - Details and the exact verdict conditions are in 03.
- `TOKENS_LIST` is a comma-separated list of experiments `inMin-inMax:outMin-outMax` (single values
  and one-sided ranges are accepted). Each entry becomes one experiment folder and one full
  stage-1/stage-2 run.
- `WORKLOAD_MIXES` is the additive alternative to `TOKENS_LIST`: bracketed mixes
  `[(profile,alpha),(profile,alpha)],[...]`, where `profile` is `inMin-inMax:outMin-outMax` and
  `alpha` is its share of the requests (normalised to 1 per mix; a repeated profile sums its
  alphas). Each bracketed entry becomes one `mix_...` experiment folder and one full
  stage-1/stage-2 run, and every request is routed to a profile by the loadgen (grammar and
  canonical rendering live in `workload_mix.py`). When it is set, `TOKENS_LIST` is ignored (the
  log says so) and `REQ_MIN_START` is consumed per mix. Malformed entries are skipped with a
  `Warning:`, never by aborting the run.

## Results tree (as implemented)

```
results/                                                      # RESULTS_DIR from .env
└── Experiment_<EXPERIMENT_TYPE>_<YYYY-MM-DD_HH-MM-SS>/       # created by _archive_execution_results()
    └── <in_min>-<in_max>_<out_min>-<out_max>/                # parent_dir per TOKENS_LIST entry
    └── mix_<label>@<alpha>+.../                              # parent_dir per WORKLOAD_MIXES entry
        └── <YYYY-MM-DD_HH-MM-SS>/                            # one folder per iteration
            ├── results.csv        # the published per-iteration summary (see 02/06)
            ├── results.json       # raw per-token events (fmperf loadgen output, copied)
            ├── output.csv         # one row per request (response time, success, success_rate)
            ├── first_half.csv / second_half.csv   # FILTER_BUFFER-trimmed halves
            └── prompts.csv        # unique prompts + token counts + occurrences
```

- The archive folder is created only at the end of the whole automation run, and it is named with
  the timestamp of that moment; the iteration folder name comes from the first response timestamp
  inside the CSVs (earliest `received_timestamp`). Additive (`WORKLOAD_MIXES`) executions use
  `Experiment_MIX_<EXPERIMENT_TYPE>_<timestamp>` as the prefix.
- Additive rows identify their experiment with `WORKLOAD_MIX` and keep the four
  `MIN/MAX_INPUT/OUTPUT_TOKENS` columns empty on purpose (see 02/03); their parent folder is the
  canonical mix (`mix_1-100_1-100@0.5+300-600_100-300@0.5`), rendered from safe characters only
  (no `:`, spaces, parentheses or slashes).
- Stale facts in `README.md` — do not copy them: the archive is actually prefixed with
  `Experiment_`, results live under `RESULTS_DIR` (not `requests/`), and for two-value
  `TOKENS_LIST` entries the parent folder is `in_out` (no ranges).
