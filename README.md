fmperf repository: https://github.com/fmperf-project/fmperf

# Setup

Set your preferences in the [.env](.env) file. Key settings:
- Duration: Set `DURATION` for each iteration length.
- Iteration cooldown: Set `ITERATION_COOLDOWN_SECONDS` to wait between iterations (default: `180`).
- Iteration hard limit: Set `ITERATION_HARD_LIMIT` to cap iterations per token-interval experiment (default: `15`). The cap is enforced when the iteration count surpasses this value.
- URL: Set `URL` to your model endpoint.
- Model discovery timeout: Set `MODEL_DISCOVERY_TIMEOUT` (seconds) for runtime model lookup via `URL/v1/models`.
- Tokens list: Set `TOKENS_LIST` as comma-separated input/output intervals. Supported formats:
	- `in:out` (single values)
	- `inMin-inMax:out` (input range, fixed output)
	- `in:outMin-outMax` (fixed input, output range)
	- `inMin-inMax:outMin-outMax` (both ranges)
	Examples: `16-64:128-256,32:64-128,64-64:256-256`.
- REQ_MIN start: Set `REQ_MIN_START` to the initial requests/min value (one value per `TOKENS_LIST` entry or per `WORKLOAD_MIXES` mix, by index; the last value is reused when the list is shorter).
- Workload mixes (additive experiments): Set `WORKLOAD_MIXES` to replace `TOKENS_LIST` with a list of mixes; each bracketed entry is one full experiment and the entries run sequentially.
	- Grammar: `[(profile,alpha),(profile,alpha)],[...]`, where `profile` is `inMin-inMax:outMin-outMax` (single values allowed, e.g. `1-100:100`) and `alpha` is that profile's share of the requests (normalised to sum 1; a profile repeated inside one mix sums its alphas).
	- Every request draws its profile with probability `alpha`, takes a prompt from that profile's own requests file, and draws its output length from that profile's output interval.
	- `MIN/MAX_INPUT/OUTPUT_TOKENS` are left empty in `results.csv` for these rows; `WORKLOAD_MIX` names the experiment and the achieved mix is reported by `ADDITIVE_EXPECTED_PROPORTIONS` / `ADDITIVE_TRUE_PROPORTIONS`.
	- Example: `WORKLOAD_MIXES=[(1-100:1-100,0.5),(300-600:100-300,0.5)],[(1-100:1-100,0.25),(300-600:100-300,0.75)]`.
- REQ_MIN increase: Set `REQ_MIN_INCREASE_MULTIPLIER` to control growth during stage 1.
- Stop threshold: Set `STOP_THRESHOLD` for the stage 2 termination criterion.

Notes:
- The values `MIN/MAX_INPUT/OUTPUT_TOKENS` are set per iteration from `TOKENS_LIST`; the [.env](.env) file is not modified during runs.
- `REQ_MIN` changes automatically per iteration based on `REQ_MIN_INCREASE_MULTIPLIER` (stage 1) and binary search (stage 2); configuration is read from [.env](.env) by [experiment_automation.py](experiment_automation.py).
- If `ITERATION_HARD_LIMIT` is exceeded in stage 1, the current token-interval experiment stops and is marked as failed.
- If `ITERATION_HARD_LIMIT` is exceeded in stage 2, the current token-interval experiment stops and returns the largest TRUE `REQ_MIN` seen in binary search.
- Generated workload files are model-agnostic: the runtime sender discovers model id from `URL/v1/models` and injects it when dispatching each request.

# How to run
Once everything is set up, run the following command:

```bash
python experiment_automation.py
```

Alternatively, if SLURM is active and running in the network, create a SLURM job to execute the experiment in the background:

```bash
sbatch experiment_automation.slurm
```

# Results

When `experiment_automation.py` finishes an execution, all results of that execution are moved into an archive folder inside the results directory (`RESULTS_DIR`) named `Experiment_[EXPERIMENT_TYPE]_[YYYY-MM-DD_HH-MM-SS]` (`Experiment_MIX_[EXPERIMENT_TYPE]_[YYYY-MM-DD_HH-MM-SS]` for `WORKLOAD_MIXES` runs). `EXPERIMENT_TYPE` is taken from the `EXPERIMENT_TYPE` environment variable (or the `.env` file), and the timestamp records when the automation finished. Each archive folder contains one subfolder per token interval, per workload mix (or per SaaS use case), as described below.

The results can be found under `RESULTS_DIR`, under the name `XXX-XXX_YYY-YYY` (`XXX` input tokens, `YYY` output tokens of the interval), or `mix_...` for a `WORKLOAD_MIXES` experiment. Within these folders, you will find more folders with the timestamp of each iteration. Finally, here you will find "first_half.csv", "second_half.csv", "output.csv", "results.csv" and "results.json". "first_half.csv" and "second_half.csv" is a summary of the response time of tokens generated in the first and second halves of the experiment, and "output.csv" is the file from which these two are obtained. "results.json" is the standard output of fmperf, where you can find information per token generated (each event of an additive run also carries its `workload_profile`). Finally, "results.csv" is a summary of the results obtained from the iteration.

Relevant `results.csv` fields:
- `REQ_MIN`: requests per minute used for the persisted iteration record. When stage 2 stops due to hard limit, this is the largest TRUE value found.
- `EVALUATION`: `TRUE` when the iteration is deemed sustainable, `FALSE` otherwise.
- `TERMINATION_REASON`: optional reason for early termination. Populated when hard-limit stop conditions are hit.
- `BINARY_SEARCH_DISTANCE`: optional absolute distance between smallest FALSE and largest TRUE values in stage 2 (`M - m`) when hard limit is exceeded.
- `BINARY_SEARCH_RELATIVE_DISTANCE`: optional relative stage-2 gap, computed as `(M - m) / (M_0 - m_0)` when hard limit is exceeded.
- `LARGEST_TRUE` / `SMALLEST_FALSE`: largest confirmed sustainable (`TRUE`) and smallest confirmed non-sustainable (`FALSE`) `REQ_MIN` values found so far. During stage 2 they are `m` and `M`, so on a hard-limit stop `SMALLEST_FALSE - LARGEST_TRUE` matches `BINARY_SEARCH_DISTANCE`. They are read after the stage update of each iteration, so for MIT an iteration whose final verdict was changed by the success-rate or plateau checks is reported under the correct column. Empty while the corresponding bound has not been confirmed yet: stage 1 confirms a FALSE only after a second consecutive FALSE at the same `REQ_MIN`, so a row with `EVALUATION=FALSE` can still show an empty `SMALLEST_FALSE` (and `LARGEST_TRUE` only lists `TRUE` values already confirmed); in stage 2 they are `m` and `M`.
- `GPU_COUNT`: number of GPUs used by the Slurm job serving the model during the iteration. Populated best-effort by `requests/store_results.py` via the shared `fmperf/utils/GpuCount.py` helper; left empty when unavailable.
- `MIN_INPUT_TOKENS` / `MAX_INPUT_TOKENS` / `MIN_OUTPUT_TOKENS` / `MAX_OUTPUT_TOKENS`: token interval evaluated in the iteration. They are empty for additive (`WORKLOAD_MIXES`) rows, whose requests belong to several profiles; the per-profile intervals of such a row live in `WORKLOAD_MIX`.
- `WORKLOAD_MIX`: canonical form of the `WORKLOAD_MIXES` entry evaluated in the iteration, e.g. `[(1-100:1-100,0.5),(300-600:100-300,0.5)]`. Empty for regular runs.
- `ADDITIVE`: `TRUE` for rows produced by an additive `WORKLOAD_MIXES` experiment, `FALSE` otherwise.
- `ADDITIVE_EXPECTED_PROPORTIONS`: JSON with the normalised `alpha` of every profile of the mix, e.g. `{"1-100:1-100":0.5,"300-600:100-300":0.5}`. Empty for regular runs.
- `ADDITIVE_TRUE_PROPORTIONS`: JSON with the observed share of each profile over all the requests of the iteration (failed requests included), to compare against the expected proportions above. Empty for regular runs.
- `RESPONSES_WITHIN_EXPECTED_INTERVAL` / `RESPONSES_OUTSIDE_EXPECTED_INTERVAL`: for additive rows each response is checked against the output interval of the profile that served it (requests that produced no tokens are excluded and counted in the log, because a failed request has no output length to classify); for regular rows every request is checked against the interval of the iteration.

Additive `WORKLOAD_MIXES` experiments follow the same stages: every mix is an experiment of its own (identified by `WORKLOAD_MIX`), `REQ_MIN_START` is taken per mix, and the thresholds and verdicts are unchanged. The only difference is the workload, because each request is routed to one of the profiles of the mix instead of belonging to a single token interval.

# How it works

The experiment consists of a set of iterations. During an iteration, of a specified duration, a number of requests will be sent per minute. The number of requests depends on the value of REQ_MIN. 

The experiment consists of two stages:

## Find non-sustainable value
REQ_MIN starts at `REQ_MIN_START` and is increased by `REQ_MIN_INCREASE_MULTIPLIER` between iterations. At the end of each iteration, we use the code in [requests/evaluate.py](requests/evaluate.py) to determine if the iteration is sustainable or not. As long as the iterations performed are sustainable, we continue to increase REQ_MIN. As soon as one of them is not sustainable, this stage ends.

If `ITERATION_HARD_LIMIT` is exceeded during this stage, the token-interval experiment ends immediately and is marked as failed in `results.csv`.

## Find MST
We set _m_ to the highest stable value for REQ_MIN and _M_ to the found unsustainable value of REQ_MIN. A binary search is performed, where REQ_MIN is set to the in-between value of _M_ and _m_ and we run an iteration. We evaluate the result, and update _m_ or _M_ accordingly, based on whether the value is deemed sustainable or not. This stage ends once _M - m_ is less than or equal to `STOP_THRESHOLD`.

If `ITERATION_HARD_LIMIT` is exceeded during this stage, the token-interval experiment ends and returns the largest TRUE value (`m`) as output. The final `results.csv` entry also records termination metadata and binary-search gap values.

## More information

Here you will find a few more pieces of information regarding how the experiment is handled:

### What is the prompt?
Prompts are loaded directly from the JSONL dataset file specified by `PROMPTS_FILE` (defaults to [oasst_roots_en_max1000_tokens.jsonl](oasst_roots_en_max1000_tokens.jsonl)). During request generation (`fmperf.loadgen.generate-input`), the prompt pool is filtered by the input token interval (`MIN_INPUT_TOKENS`/`MAX_INPUT_TOKENS`). Output token behavior remains unchanged and is sampled within `MIN_OUTPUT_TOKENS`/`MAX_OUTPUT_TOKENS`.