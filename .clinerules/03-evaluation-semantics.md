# 03 — MIT vs MST evaluation semantics

Both methods share the same stage machine (`update_stage_1`, `update_stage_2`, `end_experiment`)
and differ only in how an iteration's verdict is computed.

## Stage machine (both)

- Stage 1: `REQ_MIN` starts at `REQ_MIN_START` (picked per `TOKENS_LIST` index) and grows by
  `REQ_MIN_INCREASE_MULTIPLIER` (rounded) after a TRUE. A FALSE is only confirmed after a **second
  consecutive FALSE at the same `REQ_MIN`**; when a TRUE and a confirmed FALSE exist, the run moves
  to stage 2 with `m = highest TRUE` and `M = lowest FALSE`.
- Stage 2: binary search `REQ_MIN = (m + M) / 2`, again with double-FALSE confirmation, until
  `end_experiment` sees `M - m <= M * STOP_THRESHOLD` (relative form) → `REQ_MIN` if the last verdict
  was TRUE, otherwise `m`.
- `ITERATION_HARD_LIMIT` caps iterations per token experiment: in stage 1 the experiment is marked
  failed (`TERMINATION_REASON=FAILED_STAGE1_ITERATION_LIMIT_EXCEEDED...`, `EVALUATION=FALSE`); in
  stage 2 the largest TRUE is returned and `BINARY_SEARCH_DISTANCE` / `..._RELATIVE_DISTANCE` are
  recorded.
- Cost awareness: this `.env.example` lists 20 token combinations, so a full automation run performs
  many iterations, each consuming `DURATION` plus `ITERATION_COOLDOWN_SECONDS`.

## MIT (short, ~2 min iterations)

- `run_evaluation_pipeline` forces `evaluation_success = True` for MIT (the statistical halves in
  `evaluate.py` are not used) and treats the `0.95 * REQ_MIN` early metrics gate as
  **informational only** (`"(MIT informational only)"`).
- The real verdict is computed afterwards in `run_experiment_for_tokens` from `mit_rpm_history`
  (stage 1 only):
  1. success rate below `SUCCESS_RATE_THRESHOLD` → FALSE;
  2. throughput regression (`curr_delta < 0`) → FALSE;
  3. plateau (`abs(curr_delta) <= max(abs(prev_delta) * MIT_PLATEAU_REL_TOL, MIT_PLATEAU_ABS_TOL)`)
     → FALSE.
- The history is cleared when stage 2 starts, so stage-2 MIT iterations rely on the success-rate
  gate plus the recorded throughput.

## MST (long, ~30 min iterations)

`requests/evaluate.py` compares the two `FILTER_BUFFER`-trimmed halves of the iteration:

1. success-rate gate: any `success_rate < SUCCESS_RATE_THRESHOLD` in either half → FALSE (exit 1);
2. otherwise Welch t-test (alpha 0.1) on `complete_response_time`, plus Cohen's d, a 10 000-sample
   bootstrap CI and a TOST with `EQUIV_MARGIN = 0.02`;
3. TRUE (exit 0) when there is no statistical difference **or** the first half has the higher
   response time (i.e. the second half is not degrading).

## Additive `WORKLOAD_MIXES` experiments

An additive experiment evaluates a mix of token profiles at once. It uses the **same stage machine,
thresholds and verdicts** as a regular experiment; only the workload and the reported columns change.

- One experiment per bracketed `WORKLOAD_MIXES` entry, sequentially; `parent_dir` is the canonical
  mix (`mix_<in>_<out>@<alpha>+...`), the archive is `Experiment_MIX_<EXPERIMENT_TYPE>_<timestamp>`
  and `REQ_MIN_START[idx]` seeds the idx-th mix. `TOKENS_LIST` is ignored in this mode.
- Every request draws its profile with probability `alpha` (weighted choice in the loadgen), takes a
  prompt from that profile's own requests file and draws its output length from that profile's
  `[out_min, out_max]`. A profile file is generated once per input interval with the union of the
  output intervals of every configured profile for that input interval, so a shared file stays
  usable; the served length is always the profile's own interval.
- `MIN/MAX_INPUT/OUTPUT_TOKENS` are empty in the persisted row (the envelope is an internal detail);
  `WORKLOAD_MIX` identifies the experiment and `ADDITIVE=TRUE` marks the row.
- Mix fidelity: `ADDITIVE_EXPECTED_PROPORTIONS` (normalised alphas) vs `ADDITIVE_TRUE_PROPORTIONS`
  (observed share of all requests, failures included); both are printed to stdout.
- Interval compliance is per profile and only over requests that produced tokens; requests without
  tokens are reported as excluded and counted by `SUCCESS_RATE` instead, so a partially failing mix
  is not reported as an interval violation.
- The success-rate gate, the stage-1 `0.95 * REQ_MIN` early gate, the MIT plateau checks and the
  MST t-test/TOST are untouched: never relax them for a mix, and never let a mix verdict bypass the
  double-FALSE confirmation.

## Rules

- Apply any verdict override **before** the persistence block: `_confirmed_bounds()` is read after
  the stage update precisely so that an iteration whose verdict changed (MIT included) is stored
  under the correct `LARGEST_TRUE` / `SMALLEST_FALSE`. Both columns are **confirmed** bounds, so a
  row with `EVALUATION=FALSE` whose FALSE was not confirmed yet (first FALSE of the double-FALSE
  retry, or a stage-2 first failure) legitimately keeps an empty `SMALLEST_FALSE`. Do not relax that
  rule to fill the column: it is what keeps `SMALLEST_FALSE - LARGEST_TRUE` a valid bracket.
- Keep thresholds overridable with the existing precedence (`os.environ` → `.env` → literal default)
  and keep defaults consistent with `.env.example`. When adding a threshold, document it in
  `.env.example` **and** `README.md`.
- QoS gates are the point of MST: never relax the success-rate gate (or the 95 %-of-`REQ_MIN` early
  gate) to make an experiment pass. Lowering a bar is a configuration decision that must be visible
  in `.env.example` + README, never a silent code default.
- Known documentation gaps to fix (or to call out) rather than replicate:
  - `SUCCESS_RATE_THRESHOLD` is the variable the code reads, while `.env.example` also ships
    `SUCCESS_RATE`, which no code reads;
  - `THRESHOLD_TYPE` (`relative` / `absolute`) is documented but unimplemented — only the relative
    `M - m <= M * STOP_THRESHOLD` form exists in `end_experiment`;
  - `MIT_PLATEAU_REL_TOL` / `MIT_PLATEAU_ABS_TOL` are read from `os.environ` only and are absent
    from `.env.example`;
  - the code default for `STOP_THRESHOLD` (0.5) differs from the value shipped in `.env.example`
    (0.05).
- `evaluate.py` requires both halves; `store_results.py` tolerates their absence. Keep that
  asymmetry: a truncated iteration can legitimately have no halves.
- Client-side timeouts (`REQUEST_TIMEOUT`, `TTFT_TIMEOUT`, `TPOT_TIMEOUT`) surface as failed
  requests and therefore as success-rate loss. Keep them comfortably above the expected response
  time of the target token interval, and remember they are per-request/per-token guards, not
  iteration guards.
