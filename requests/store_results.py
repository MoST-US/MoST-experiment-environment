import os
import re
import csv
import json
import math
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# Best-effort GPU_COUNT resolution. fmperf may not be importable when this
# script runs as a standalone subprocess; degrade gracefully in that case.
try:
    _FMPERF_ROOT = Path(__file__).resolve().parent.parent
    if str(_FMPERF_ROOT) not in sys.path:
        sys.path.insert(0, str(_FMPERF_ROOT))
    from fmperf.utils.GpuCount import GPU_COUNT_FIELD, find_model_job_gpu_count
except Exception:
    GPU_COUNT_FIELD = "GPU_COUNT"

    def find_model_job_gpu_count(*_args, **_kwargs):
        raise RuntimeError("fmperf.utils.GpuCount is unavailable")

# Best-effort helpers to enrich results.csv with requested fields
def _read_env_value(env_path: Path, key: str, default: str = "") -> str:
    try:
        if env_path.exists():
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if line.startswith(key + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return default


def _normalize_endpoint_value(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def _normalize_endpoint_base_url(value: str | None) -> str:
    """Normalize endpoint into a base URL like http://host:port.

    Accepts values with or without scheme, e.g.:
    - gpu05:9000
    - http://gpu05:9000
    - http://gpu05:9000/v1/completions
    """
    endpoint = _normalize_endpoint_value(value)
    if not endpoint:
        return ""

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    # Typical env value without scheme.
    if " " in endpoint or endpoint.startswith("/"):
        return ""
    if ":" in endpoint:
        parsed2 = urllib.parse.urlparse(f"http://{endpoint}")
        if parsed2.netloc:
            return urllib.parse.urlunparse((parsed2.scheme, parsed2.netloc, "", "", "", ""))

    return ""


def _is_endpoint_like(value: str | None) -> bool:
    return bool(_normalize_endpoint_base_url(value))

def _parse_node_port(url: str) -> tuple[str, str] | None:
    """Return (node, port) from an endpoint like gpu05:9000 or http://gpu05:9000/v1."""
    endpoint = _normalize_endpoint_value(url)
    if not endpoint:
        return None

    parsed = urllib.parse.urlparse(endpoint)
    hostport = parsed.netloc if (parsed.scheme and parsed.netloc) else endpoint
    hostport = hostport.split("/", 1)[0]
    if ":" not in hostport:
        return None

    host, port = hostport.rsplit(":", 1)
    host = host.strip()
    port = port.strip()
    if not host or not port.isdigit():
        return None
    return host, port


def _resolve_gpu_count(model_used: str, url: str) -> str:
    """Best-effort GPU_COUNT (str) for the model-serving Slurm job; "" when unavailable."""
    if not model_used or not url:
        return ""
    node_port = _parse_node_port(url)
    if node_port is None:
        return ""
    node, port = node_port
    try:
        result = find_model_job_gpu_count(model_used, node, port)
        return str(result["gpuCount"])
    except Exception:
        return ""

def _find_slurm_log(job_id: str | None) -> tuple[str | None, str | None]:
    """Return (job_id, slurm_log_path) if found.
    Strategy:
    - Prefer explicit job: look for slurm-<jobid>.out
      in likely roots: CWD, script dir, and script dir's parent (project root).
    - If not found, fallback to most recent slurm-*.out among those roots
      and their parents up to a few levels.
    """
    job = job_id or os.environ.get("SLURM_JOB_ID")

    candidates: list[Path] = []

    # Build a robust search set of directories
    roots: list[Path] = []
    try:
        cwd = Path.cwd()
        roots.extend([cwd, *cwd.parents[:4]])
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        script_dir = here.parent
        roots.append(script_dir)
        # If this file lives in 'requests/', the project root is its parent
        if script_dir.name.lower() == "requests":
            roots.append(script_dir.parent)
        roots.extend([*script_dir.parents[:4]])
    except Exception:
        pass

    # De-duplicate while preserving order
    seen = set()
    unique_roots: list[Path] = []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if rp not in seen:
            seen.add(rp)
            unique_roots.append(rp)

    # First pass: exact slurm-<job>.out in likely dirs
    if job:
        try:
            for up in unique_roots:
                p = up / f"slurm-{job}.out"
                if p.exists():
                    return job, str(p)
        except Exception:
            pass

    # Fallback: collect all slurm-*.out files in the search dirs
    try:
        for up in unique_roots:
            for entry in up.glob("slurm-*.out"):
                candidates.append(entry)
    except Exception:
        pass

    if candidates:
        try:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            picked = candidates[0]
            m = re.search(r"slurm-(\d+)\.out$", picked.name)
            jid = m.group(1) if m else (job or None)
            return jid, str(picked)
        except Exception:
            pass

    return job, None


def _query_model_id_from_endpoint(value: str | None, timeout_seconds: float = 10.0) -> str:
    """Query /v1/models and return the first model id, mirroring run.py behavior."""
    base_url = _normalize_endpoint_base_url(value)
    if not base_url:
        return ""

    models_url = urllib.parse.urljoin(base_url + "/", "v1/models")
    try:
        req = urllib.request.Request(
            models_url,
            headers={"Accept": "application/json", "User-Agent": "store-results/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                return ""
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return ""
    except Exception:
        return ""

    if not isinstance(payload, dict):
        return ""
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return ""
    first = data[0]
    if not isinstance(first, dict):
        return ""
    model_id = first.get("id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    return ""


def _resolve_model_used_fresh(
    resolved_model_cli: str,
    model_arg: str,
    endpoint_candidates: list[str],
    timeout_seconds: float = 10.0,
) -> str:
    """Resolve MODEL_USED from deterministic sources only.

    Order:
    1) A resolved model provided by caller (if not endpoint-like).
    2) Query endpoint(s) via /v1/models and wait for response.
    3) Use MODEL arg only when it is already a model id (not endpoint-like).
    """
    cli_model = (resolved_model_cli or "").strip()
    if cli_model and not _is_endpoint_like(cli_model):
        return cli_model

    query_targets: list[str] = []
    for endpoint in endpoint_candidates:
        normalized = _normalize_endpoint_value(endpoint)
        if normalized:
            query_targets.append(normalized)

    model_raw = (model_arg or "").strip()
    if model_raw and _is_endpoint_like(model_raw):
        query_targets.append(model_raw)

    for endpoint in query_targets:
        resolved = _query_model_id_from_endpoint(endpoint, timeout_seconds=timeout_seconds)
        if resolved:
            return resolved

    if model_raw and not _is_endpoint_like(model_raw):
        return model_raw

    return ""

def _extract_median_tokens_from_log(slurm_path: str | None) -> str | None:
    """Parse the latest 'Median tokens per response: <value>' printed by experiment_automation.
    Prefer the last occurrence in the Slurm log; return None if unavailable.
    """
    if not slurm_path:
        return None
    try:
        last_val: str | None = None
        pat = re.compile(r"Median tokens per response:\s*([0-9]+(?:\.[0-9]+)?)")
        with open(slurm_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    last_val = m.group(1)
        return last_val
    except Exception:
        return None

def _to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _to_int(value) -> int | None:
    fv = _to_float(value)
    if fv is None:
        return None
    try:
        return int(fv)
    except Exception:
        return None


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if math.isfinite(v) and float(v).is_integer():
        return str(int(v))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _percentile(values_sorted: list[float], percentile: float) -> float | None:
    if not values_sorted:
        return None
    if len(values_sorted) == 1:
        return values_sorted[0]
    p = max(0.0, min(100.0, float(percentile)))
    pos = (p / 100.0) * (len(values_sorted) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values_sorted[lo]
    frac = pos - lo
    return values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * frac


def _serialize_percentiles(values: list[float]) -> str:
    if not values:
        return ""
    vals = sorted(values)
    pts = {
        "p50": _percentile(vals, 50),
        "p75": _percentile(vals, 75),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
    }
    clean = {k: float(_format_number(v)) for k, v in pts.items() if v is not None}
    return json.dumps(clean, separators=(",", ":"), ensure_ascii=True)


def _load_prompt_tokens(filename: str | None = None) -> dict[int, float]:
    """Prompt token counts of a requests file, keyed by the position inside that file.

    `filename` selects the requests file of a workload profile (additive runs); None uses
    REQUESTS_FILENAME, i.e. the single workload file of a regular run.
    """
    out: dict[int, float] = {}
    req_path = _find_requests_file(filename)
    if not req_path:
        return out
    try:
        payload = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    if isinstance(payload, dict):
        items = payload.get("requests") or payload.get("data") or payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        raw = it.get("prompt_token_count")
        if raw is None:
            raw = it.get("input_token_count")
        if raw is None:
            raw = it.get("prompt_len")
        if raw is None:
            raw = it.get("input_tokens")
        if raw is None and isinstance(it.get("config"), dict):
            raw = it["config"].get("in_tokens")
        fv = _to_float(raw)
        if fv is not None and fv >= 0:
            out[idx] = fv
    return out


def _is_additive_run() -> bool:
    """True when the iteration belongs to an additive WORKLOAD_MIXES experiment.

    The automation exports ADDITIVE through the environment (never argv). Additive rows keep
    MIN/MAX_INPUT/OUTPUT_TOKENS empty and describe the experiment with WORKLOAD_MIX instead.
    """
    return os.environ.get('ADDITIVE', '').strip().upper() in ('TRUE', '1', 'YES')


def _load_workload_profiles() -> dict[str, dict]:
    """Map each WORKLOAD_MIX_SPEC profile label to its output interval and requests file.

    Duplicated on purpose, like the other environment readers in requests/*.py: this script runs
    standalone and resolves everything from os.environ plus argv. Returns {} when unavailable.
    """
    raw = (os.environ.get('WORKLOAD_MIX_SPEC') or '').strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(f"Warning: unable to parse WORKLOAD_MIX_SPEC: {exc}")
        return {}
    raw_profiles = payload.get('profiles') if isinstance(payload, dict) else None
    if not isinstance(raw_profiles, list):
        print("Warning: WORKLOAD_MIX_SPEC has no 'profiles' list; per-profile data unavailable.")
        return {}
    profiles: dict[str, dict] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            continue
        label = str(raw_profile.get('label') or '').strip()
        if not label:
            continue
        profiles[label] = {
            'out_min': _to_float(raw_profile.get('out_min')),
            'out_max': _to_float(raw_profile.get('out_max')),
            'alpha': _to_float(raw_profile.get('alpha')),
            'filename': str(raw_profile.get('filename') or '').strip(),
        }
    return profiles


def _additive_request_filenames() -> list[str] | None:
    """Unique requests files of the current additive run (None when the run is not additive).

    An empty list means "additive, but no usable WORKLOAD_MIX_SPEC": callers then leave the
    additive-only outputs empty instead of falling back to the (non-existent) envelope file.
    """
    if not _is_additive_run():
        return None
    filenames: list[str] = []
    for info in _load_workload_profiles().values():
        filename = info.get('filename') or ''
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def _additive_expected_proportions(workload_profiles: dict[str, dict]) -> str:
    """JSON of the normalised expected proportion of each profile ('' when unavailable)."""
    weights = {label: (info.get('alpha') or 0.0) for label, info in workload_profiles.items()}
    total = sum(weights.values())
    if total <= 0:
        return ''
    expected = {label: round(weight / total, 6) for label, weight in weights.items()}
    return json.dumps(expected, separators=(',', ':'), ensure_ascii=True)


def _additive_true_proportions(
    profile_labels: list[str | None], workload_profiles: dict[str, dict]
) -> str:
    """JSON of the observed proportion of each profile over all counted requests ('' if none).

    Requests are counted whether they succeeded or not, so the observed proportions are comparable
    with the expected alphas of the mix (a failing profile still consumes its share of the
    scheduled requests).
    """
    if not profile_labels or not workload_profiles:
        return ''
    counts = {label: 0 for label in workload_profiles}
    for label in profile_labels:
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    if total <= 0:
        return ''
    observed = {label: round(count / total, 6) for label, count in counts.items()}
    return json.dumps(observed, separators=(',', ':'), ensure_ascii=True)


def _load_prompt_tokens_by_profile(workload_profiles: dict[str, dict]) -> dict[str, dict[int, float]]:
    """Prompt token counts per profile label, resolved through each profile's requests file.

    Profiles sharing a requests file share the loaded mapping (each file is read only once).
    """
    by_profile: dict[str, dict[int, float]] = {}
    cache: dict[str, dict[int, float]] = {}
    for label, info in workload_profiles.items():
        filename = info.get('filename') or ''
        if not filename:
            continue
        if filename not in cache:
            cache[filename] = _load_prompt_tokens(filename)
        by_profile[label] = cache[filename]
    return by_profile


def _load_requests_items(req_path: Path) -> list:
    """Request entries of a requests JSON file (accepts a list or a wrapped payload)."""
    try:
        payload = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: unable to parse requests from {req_path}: {e}")
        return []
    if isinstance(payload, dict):
        return payload.get("requests") or payload.get("data") or payload.get("items") or []
    if isinstance(payload, list):
        return payload
    return []


def _prompt_entries(items) -> list[tuple[str, Optional[str]]]:
    """(prompt_text, prompt_token_count) of each request entry, in file order."""
    prompts_by_index: list[tuple[str, Optional[str]]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        txt = it.get("prompt_text")
        tok = it.get("prompt_token_count")
        if not (isinstance(txt, str) and txt.strip()):
            continue
        tok_str: Optional[str] = None
        if isinstance(tok, (int, float)):
            tok_str = str(int(tok)) if isinstance(tok, int) or float(tok).is_integer() else str(tok)
        elif isinstance(tok, str) and tok.strip():
            tok_str = tok.strip()
        prompts_by_index.append((txt, tok_str))
    return prompts_by_index


def _prompt_token_values(req_path: Path) -> list[float]:
    """Numeric prompt_token_count values of a requests file."""
    vals: list[float] = []
    for it in _load_requests_items(req_path):
        if not isinstance(it, dict):
            continue
        fv = _to_float(it.get("prompt_token_count"))
        if fv is not None and fv >= 0:
            vals.append(fv)
    return vals


def _parse_numeric_bounds(min_value: str | None, max_value: str | None) -> tuple[float | None, float | None]:
    lo = _to_float(min_value)
    hi = _to_float(max_value)
    if lo is None and hi is None:
        return None, None
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _compute_request_token_stats(
    results_path: Path,
    expected_min_output: float | None,
    expected_max_output: float | None,
    workload_profiles: dict[str, dict] | None = None,
) -> dict[str, str | None]:
    """Compute output/input token statistics and interval compliance from results.json.

    workload_profiles maps each profile label of an additive run (WORKLOAD_MIX_SPEC) to its output
    interval, alpha and requests file. When given, every request is validated against the interval
    of the profile that served it instead of the single mix-envelope interval, and the observed
    proportion of each profile is reported next to the expected one. Requests that produced no
    tokens are not interval violations (they surface in SUCCESS_RATE) and are excluded from the
    compliance counts, so a partially failing mix does not look like an interval violation.
    """
    out: dict[str, str | None] = {
        "median_response_tokens": None,
        "total_requests": None,
        "success_rate": None,
        "responses_within_interval": None,
        "responses_outside_interval": None,
        "additive_expected_proportions": None,
        "additive_true_proportions": None,
        "avg_tokens_per_request": None,
        "avg_tokens_per_response": None,
        "input_token_variance": None,
        "output_token_variance": None,
        "input_token_percentiles": None,
        "output_token_percentiles": None,
        "request_total_token_percentiles": None,
    }

    # success rate from output.csv if present
    try:
        out_csv = Path("output.csv")
        if out_csv.exists():
            with out_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row:
                    col = None
                    for c in ("success_rate", "success_ratio", "success", "pass_rate", "accuracy"):
                        if c in row:
                            col = c
                            break
                    if not col:
                        for k in row.keys():
                            if "success" in k.lower():
                                col = k
                                break
                    if col:
                        out["success_rate"] = str(row.get(col, "") or "")
    except Exception:
        pass

    request_entries: dict[tuple[int | None, int | None], dict[str, float | int | None]] = {}

    # Compute request-level output lengths from results.json if available
    try:
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                items = data["results"]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            current_rid = -1
            for it in items:
                if not isinstance(it, dict):
                    continue

                rid = _to_int(it.get("request_idx"))
                wid = _to_int(it.get("worker_idx"))
                rsi = _to_int(it.get("response_idx"))
                n_tokens = _to_float(it.get("n_tokens"))
                sample_idx = _to_int(it.get("sample_idx"))
                raw_profile = it.get("workload_profile")
                profile_label = (
                    raw_profile.strip()
                    if isinstance(raw_profile, str) and raw_profile.strip()
                    else None
                )

                # Fallback request derivation when request_idx is missing.
                if rid is None:
                    if rsi is not None and rsi == 0:
                        current_rid += 1
                    if current_rid < 0:
                        current_rid = 0
                    rid = current_rid

                key = (wid, rid)
                if key not in request_entries:
                    request_entries[key] = {
                        "token_sum": 0.0,
                        "max_response_idx": -1,
                        "sample_idx": sample_idx,
                        # Additive runs tag every event with its workload profile; None for regular
                        # runs and for events that carry no label.
                        "profile": profile_label,
                        "has_tokens": False,
                    }

                rec = request_entries[key]
                if sample_idx is not None and rec.get("sample_idx") is None:
                    rec["sample_idx"] = sample_idx
                if profile_label and not rec.get("profile"):
                    rec["profile"] = profile_label
                if n_tokens is not None and n_tokens > 0:
                    rec["token_sum"] = float(rec.get("token_sum", 0.0) or 0.0) + n_tokens
                    # Requests that really produced tokens: without them the request length below
                    # is only an estimate (max_response_idx + 1), which must not be treated as a
                    # served response by the additive interval validation.
                    rec["has_tokens"] = True
                if rsi is not None:
                    prev = int(rec.get("max_response_idx", -1) or -1)
                    if rsi > prev:
                        rec["max_response_idx"] = rsi
    except Exception:
        pass

    output_tokens_per_request: list[float] = []
    request_sample_indices: list[int | None] = []
    request_profiles: list[str | None] = []
    request_has_tokens: list[bool] = []
    if request_entries:
        for rec in request_entries.values():
            token_sum = _to_float(rec.get("token_sum")) or 0.0
            max_idx = _to_int(rec.get("max_response_idx"))
            if token_sum > 0:
                out_len = token_sum
            elif max_idx is not None and max_idx >= 0:
                out_len = float(max_idx + 1)
            else:
                out_len = 0.0
            output_tokens_per_request.append(out_len)
            request_sample_indices.append(_to_int(rec.get("sample_idx")))
            request_profiles.append(rec.get("profile") if isinstance(rec.get("profile"), str) else None)
            request_has_tokens.append(bool(rec.get("has_tokens")))

    if output_tokens_per_request:
        sorted_out = sorted(output_tokens_per_request)
        out["median_response_tokens"] = _format_number(_percentile(sorted_out, 50))
        out["total_requests"] = str(len(output_tokens_per_request))
        out["avg_tokens_per_response"] = _format_number(sum(output_tokens_per_request) / len(output_tokens_per_request))
        out["output_token_variance"] = _format_number(_population_variance(output_tokens_per_request))
        out["output_token_percentiles"] = _serialize_percentiles(output_tokens_per_request)

        if workload_profiles:
            # Additive run: validate every request against the interval of the profile that served
            # it. Requests without a label or without tokens are not interval violations; they are
            # counted separately so a mis-wired mix is visible in the log instead of silently
            # inflating the violation count.
            within = 0
            outside = 0
            unlabeled = 0
            without_tokens = 0
            violations_by_profile: dict[str, int] = {}
            for position, out_len in enumerate(output_tokens_per_request):
                label = request_profiles[position] if position < len(request_profiles) else None
                info = workload_profiles.get(label) if label else None
                if info is None:
                    unlabeled += 1
                    continue
                if not (request_has_tokens[position] if position < len(request_has_tokens) else False):
                    without_tokens += 1
                    continue
                low = info.get("out_min")
                high = info.get("out_max")
                if (low is None or out_len >= low) and (high is None or out_len <= high):
                    within += 1
                else:
                    outside += 1
                    violations_by_profile[label] = violations_by_profile.get(label, 0) + 1
            out["responses_within_interval"] = str(within)
            out["responses_outside_interval"] = str(outside)
            if unlabeled:
                print(
                    f"Warning: {unlabeled} request(s) had no 'workload_profile' label and were "
                    "excluded from RESPONSES_WITHIN_EXPECTED_INTERVAL / "
                    "RESPONSES_OUTSIDE_EXPECTED_INTERVAL."
                )
            if without_tokens:
                print(
                    f"Warning: {without_tokens} request(s) produced no tokens and were excluded from "
                    "the interval compliance columns (see SUCCESS_RATE)."
                )
            if violations_by_profile:
                print(
                    "Warning: additive output tokens outside the profile interval "
                    f"(violations per profile): {violations_by_profile}"
                )
            out["additive_expected_proportions"] = _additive_expected_proportions(workload_profiles)
            out["additive_true_proportions"] = _additive_true_proportions(
                request_profiles, workload_profiles
            )
        elif expected_min_output is not None or expected_max_output is not None:
            within = 0
            for out_len in output_tokens_per_request:
                ok_lo = expected_min_output is None or out_len >= expected_min_output
                ok_hi = expected_max_output is None or out_len <= expected_max_output
                if ok_lo and ok_hi:
                    within += 1
            outside = len(output_tokens_per_request) - within
            out["responses_within_interval"] = str(within)
            out["responses_outside_interval"] = str(outside)

    # If results.json unavailable, attempt to derive total from first/second_half.csv
    if out["total_requests"] is None:
        try:
            fh = Path("first_half.csv")
            sh = Path("second_half.csv")
            count = 0
            for p in (fh, sh):
                if p.exists():
                    with p.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        if rows:
                            count += max(0, len(rows) - 1)
            if count:
                out["total_requests"] = str(count)
        except Exception:
            pass

    if workload_profiles:
        # Additive run: every request is measured with the prompts of its own profile file.
        prompt_tokens_by_idx: dict[int, float] = {}
        prompt_tokens_by_profile = _load_prompt_tokens_by_profile(workload_profiles)
    else:
        prompt_tokens_by_idx = _load_prompt_tokens()
        prompt_tokens_by_profile = {}
    if request_sample_indices and (prompt_tokens_by_idx or prompt_tokens_by_profile):
        input_tokens_per_request: list[float] = []
        total_tokens_per_request: list[float] = []
        for i, sample_idx in enumerate(request_sample_indices):
            if sample_idx is None:
                continue
            if prompt_tokens_by_profile:
                label = request_profiles[i] if i < len(request_profiles) else None
                input_tokens = (prompt_tokens_by_profile.get(label or "") or {}).get(sample_idx)
            else:
                input_tokens = prompt_tokens_by_idx.get(sample_idx)
            if input_tokens is None:
                continue
            input_tokens_per_request.append(input_tokens)
            if i < len(output_tokens_per_request):
                total_tokens_per_request.append(input_tokens + output_tokens_per_request[i])

        if input_tokens_per_request:
            out["input_token_variance"] = _format_number(_population_variance(input_tokens_per_request))
            out["input_token_percentiles"] = _serialize_percentiles(input_tokens_per_request)
        if total_tokens_per_request:
            out["avg_tokens_per_request"] = _format_number(sum(total_tokens_per_request) / len(total_tokens_per_request))
            out["request_total_token_percentiles"] = _serialize_percentiles(total_tokens_per_request)

    # Fallback for avg tokens/request if input tokens are not available.
    if out["avg_tokens_per_request"] is None:
        out["avg_tokens_per_request"] = out["avg_tokens_per_response"]

    return out

_TOKEN_RANGE_RE = re.compile(r"^\d+(?:-\d+)?$")


def _looks_like_token_range(value: str | None) -> bool:
    """True when the value looks like a token interval, e.g. '128' or '128-256'."""
    return bool(_TOKEN_RANGE_RE.match(str(value).strip()))


def _compact_layout_signature(args: list[str], offset: int = 0, saas_mode: bool = False) -> bool:
    """True when args[offset:] matches the compact CLI layout built by experiment_automation.py.

    Compact layout (offset 0): model, stage, parent_dir, in_range, out_range, req_min, evaluation.
    The signature is positive - stage in {1, 2}, non-empty parent_dir, token intervals in the
    in/out positions (relaxed in SaaS mode, where those slots carry the use-case id) and a
    TRUE/FALSE evaluation - so it cannot be satisfied by the legacy layout, whose offset-1 slot
    holds the GPU count, offset-3 the node name and offset-6 a token range.
    """
    if len(args) < offset + 7:
        return False
    intervals_ok = saas_mode or (
        _looks_like_token_range(args[offset + 3]) and _looks_like_token_range(args[offset + 4])
    )
    return (
        str(args[offset + 1]).strip() in ("1", "2")
        and bool(str(args[offset + 2]).strip())
        and intervals_ok
        and str(args[offset + 6]).strip().upper() in ("TRUE", "FALSE")
    )


def _parse_range(value: str | None) -> tuple[str | None, str | None]:
    """Parse a token range string like '32-64' or '32' into (min,max) strings.
    Returns (None, None) if input is falsy.
    """
    if value is None:
        return None, None
    s = str(value).strip()
    if not s:
        return None, None
    if '-' in s:
        a, b = s.split('-', 1)
        return a.strip() or None, b.strip() or None
    return s, s

def _compute_median_prompt_tokens() -> str | None:
    """Compute median of prompt_token_count from the generated requests file(s).

    Additive runs (WORKLOAD_MIX_SPEC) take the median over the prompts of every profile file, each
    file counted once even when several profiles share it. Falls back to None if unavailable or
    unparseable.
    """
    try:
        additive_filenames = _additive_request_filenames()
        filenames: list[str | None] = (
            [None] if additive_filenames is None else list(additive_filenames)
        )
        vals: list[float] = []
        for filename in filenames:
            req_path = _find_requests_file(filename)
            if not req_path:
                continue
            vals.extend(_prompt_token_values(req_path))

        if not vals:
            return None
        vals.sort()
        n = len(vals)
        if n % 2 == 1:
            return f"{vals[n//2]:.0f}" if vals[n//2].is_integer() else f"{vals[n//2]:.3f}"
        m = (vals[n//2 - 1] + vals[n//2]) / 2.0
        return f"{m:.0f}" if m.is_integer() else f"{m:.3f}"
    except Exception:
        return None

def _find_requests_file(filename: str | None = None) -> Optional[Path]:
    """Locate the requests JSON file generated by generate-input.

    Strategy:
    - Use REQUESTS_FILENAME/REQUESTS_DIR from env if available.
    - Try common relative locations from current dir and its parent.
    - Return the first existing Path, else None.

    `filename` overrides the environment value: every workload profile of an additive run has its
    own requests file (see WORKLOAD_MIX_SPEC).
    """
    fname = filename if filename else os.environ.get("REQUESTS_FILENAME", None)
    rdir = os.environ.get("REQUESTS_DIR", None)

    candidates: list[Path] = []

    # Direct filename in CWD
    if fname:
        candidates.append(Path(fname))
    # REQUESTS_DIR + filename
    if fname and rdir:
        candidates.append(Path(rdir) / fname)
    # Parent dir + filename
    if fname:
        candidates.append(Path("..") / fname)
    # Parent dir + REQUESTS_DIR + filename
    if fname and rdir:
        candidates.append(Path("..") / rdir / fname)

    # Common defaults
    candidates.extend([
        Path("requests.json"),
        Path("..") / "requests.json",
        Path("requests") / "requests.json",
        Path("..") / "requests" / "requests.json",
    ])

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

def _write_prompts_csv(full_dir_path: str) -> None:
    """Create prompts.csv with unique prompts used in this run.

    Reads the requests JSON produced by generate-input (cases with
    'prompt_text' and optional 'prompt_token_count') and writes a CSV
    containing each unique prompt aggregated with occurrences. Additive runs
    (WORKLOAD_MIX_SPEC) aggregate the prompts of every profile file, each file
    counted once even when several profiles share it.

    Occurrences reflect how many times the prompt appeared in the
    requests payload (i.e., how many requests were sent using that
    prompt), independent of response success.
    """
    try:
        additive_filenames = _additive_request_filenames()
        if additive_filenames is None:
            req_path = _find_requests_file()
            if not req_path:
                print("Warning: requests file not found; skipping prompts.csv generation")
                return
            prompts_by_index = _prompt_entries(_load_requests_items(req_path))
        else:
            prompts_by_index = []
            for filename in additive_filenames:
                req_path = _find_requests_file(filename)
                if not req_path:
                    print(f"Warning: requests file not found: {filename}")
                    continue
                prompts_by_index.extend(_prompt_entries(_load_requests_items(req_path)))

        if not prompts_by_index:
            print("Warning: no prompts found in requests payload; skipping prompts.csv generation")
            return

        # Aggregate by prompt text: count occurrences
        from collections import OrderedDict
        agg: "OrderedDict[str, dict]" = OrderedDict()
        for idx, (txt, tok) in enumerate(prompts_by_index):
            if txt not in agg:
                agg[txt] = {"count": 0, "token_count": tok}
            entry = agg[txt]
            entry["count"] += 1
            if entry["token_count"] in (None, "") and tok not in (None, ""):
                entry["token_count"] = tok

        out_path = os.path.join(full_dir_path, "prompts.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["PROMPT_TEXT", "PROMPT_TOKEN_COUNT", "OCCURRENCES"])
            for txt, entry in agg.items():
                safe_txt = txt.replace("\n", " ").strip()
                w.writerow([
                    safe_txt,
                    "" if entry["token_count"] in (None, "") else str(entry["token_count"]),
                    str(entry["count"]),
                ])
        print(f"Created prompts.csv in {full_dir_path} with {len(agg)} unique prompts")
    except Exception as e:
        print(f"Warning: failed to create prompts.csv: {e}")


_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%d/%m/%Y  %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y%m%dT%H%M%S",
)


def _parse_timestamp_string(value: str | None) -> datetime | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _extract_timestamp_from_csv(path: Path) -> tuple[datetime | None, str | None]:
    if not path.exists():
        return None, None
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # header
            row = next(reader, None)
    except Exception:
        return None, None
    if not row:
        return None, None
    raw_value = row[0] if row else None
    return _parse_timestamp_string(raw_value), raw_value


def _sanitize_dir_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def _derive_directory_name() -> tuple[str, str]:
    candidates = [
        Path("first_half.csv"),
        Path("second_half.csv"),
        Path("output.csv"),
    ]
    for candidate in candidates:
        dt, raw = _extract_timestamp_from_csv(candidate)
        if dt:
            return dt.strftime("%Y-%m-%d_%H-%M-%S"), candidate.name
        if raw:
            return _sanitize_dir_component(raw), candidate.name
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S"), "current_time"


def _resolve_results_dir() -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / '.env'
    results_dir = os.environ.get('RESULTS_DIR') or _read_env_value(env_path, 'RESULTS_DIR', 'results')
    p = Path(results_dir)
    if not p.is_absolute():
        p = root_dir / p
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    try:
        results_dir = _resolve_results_dir()
        os.chdir(results_dir)

        # Additive (WORKLOAD_MIXES) iterations are identified by ADDITIVE plus WORKLOAD_MIX_SPEC,
        # which the automation exports through the environment (never argv, see
        # experiment_automation.py).
        additive = _is_additive_run()
        workload_mix = os.environ.get('WORKLOAD_MIX', '').strip()
        workload_profiles = _load_workload_profiles() if additive else {}
        if additive:
            print(f"Additive run detected: WORKLOAD_MIX={workload_mix or '<unknown>'}")
            if not workload_profiles:
                print(
                    "Warning: ADDITIVE is set but WORKLOAD_MIX_SPEC has no usable profiles; "
                    "the per-profile interval validation and the proportions are disabled."
                )

        # Check if required CSV files exist
        if not os.path.exists("output.csv"):
            print("Error: output.csv not found")
            return

        missing_halves = [name for name in ("first_half.csv", "second_half.csv") if not os.path.exists(name)]
        if missing_halves:
            print(f"Warning: {', '.join(missing_halves)} not found; proceeding without filtered halves.")
        
        # Determine directory name based on the first available timestamp source
        dir_name, timestamp_source = _derive_directory_name()
        print(f"Using timestamp from {timestamp_source} to create directory: {dir_name}")
        
        # Parse CLI arguments from experiment_automation.py.
        # Supported formats:
        # - Compact (current): model, stage, parent_dir, in_range, out_range, req_min, evaluation, [median]
        # - Legacy:            model, gpus, cpus, node, stage, parent_dir, in_range, out_range, req_min, evaluation, [median]
        args = sys.argv[1:]
        # Layout detection. The compact layout is the one experiment_automation.py builds:
        #   model, stage, parent_dir, in_range, out_range, req_min, evaluation, ...
        # It used to be detected with '"_" in args[2]', which silently fell back to the legacy
        # layout whenever a leading argument was added; every column was then read one position
        # off (parent_dir -> out_range, LARGEST_TRUE -> finished flag, SMALLEST_FALSE never set).
        # Use a positive signature and always log the layout that was selected.
        saas_mode = os.environ.get("SERVICE_TYPE", "").strip() == "SaaS"
        is_compact_cli = _compact_layout_signature(args, 0, saas_mode)
        # Offset 3 also matches the legacy layout, so only offsets 1-2 are unambiguous evidence
        # of the compact layout being pushed down by extra leading arguments.
        misplaced_compact_offset = None
        if not _compact_layout_signature(args, 3, saas_mode):
            misplaced_compact_offset = next(
                (
                    offset
                    for offset in (1, 2)
                    if _compact_layout_signature(args, offset, saas_mode)
                ),
                None,
            )
        if is_compact_cli:
            print(
                "CLI format: compact (model, stage, parent_dir, in_range, out_range, req_min, "
                f"evaluation, ...) [{len(args)} args]"
            )
        elif misplaced_compact_offset is not None:
            print(
                f"Warning: compact CLI layout detected at argument index {misplaced_compact_offset} "
                f"instead of 0; every results.csv column is shifted by {misplaced_compact_offset} "
                "position(s). Append new arguments at the end of store_args in "
                "experiment_automation.py, never in the middle."
            )
            if len(args) >= 10:
                print(
                    "CLI format: legacy fallback (compact layout found at offset "
                    f"{misplaced_compact_offset}) [{len(args)} args]"
                )
            else:
                print(
                    "CLI format: environment fallback (compact layout found at offset "
                    f"{misplaced_compact_offset}) [{len(args)} args]"
                )
        elif len(args) >= 10:
            print(
                f"CLI format: legacy (model, gpus, cpus, node, stage, parent_dir, ...) [{len(args)} args]"
            )
        else:
            print(
                f"CLI format: environment/no-args (values read from os.environ) [{len(args)} args]"
            )
        experiment_type = os.environ.get('EXPERIMENT_TYPE', '')
        parent_dir = None
        model = os.environ.get('MODEL', '')
        stage = os.environ.get('STAGE', '')
        
        # Resolve tokens (min/max), REQ_MIN, EVALUATION, MEDIAN: prefer CLI args, then env/log, then .env
        min_input_tokens = ''
        max_input_tokens = ''
        min_output_tokens = ''
        max_output_tokens = ''
        req_min = ''
        evaluation_flag = ''
        median_cli = ''
        resolved_model_cli = ''
        termination_reason = ''
        binary_distance_abs = ''
        binary_distance_rel = ''
        largest_true = ''
        smallest_false = ''
        finished_flag = ''

        if is_compact_cli:
            model = args[0]
            stage = args[1]
            parent_dir = args[2]

            in_range_str = args[3]
            out_range_str = args[4]
            mi, ma = _parse_range(in_range_str)
            mo, moa = _parse_range(out_range_str)
            min_input_tokens = mi or ''
            max_input_tokens = ma or ''
            min_output_tokens = mo or ''
            max_output_tokens = moa or ''
            req_min = args[5]
            evaluation_flag = args[6]
            if len(args) >= 8:
                median_cli = args[7]
            if len(args) >= 9:
                resolved_model_cli = args[8]
            if len(args) >= 10:
                termination_reason = args[9]
            if len(args) >= 11:
                binary_distance_abs = args[10]
            if len(args) >= 12:
                binary_distance_rel = args[11]
            if len(args) >= 13:
                largest_true = args[12]
            if len(args) >= 14:
                smallest_false = args[13]
            if len(args) >= 15:
                finished_flag = args[14]
        elif len(args) >= 10:
            # Backward-compatible parsing for legacy positional arguments.
            model = args[0]
            stage = args[4]
            parent_dir = args[5]

            in_range_str = args[6]
            out_range_str = args[7]
            mi, ma = _parse_range(in_range_str)
            mo, moa = _parse_range(out_range_str)
            min_input_tokens = mi or ''
            max_input_tokens = ma or ''
            min_output_tokens = mo or ''
            max_output_tokens = moa or ''
            req_min = args[8]
            evaluation_flag = args[9]
            if len(args) >= 11:
                median_cli = args[10]
            if len(args) >= 12:
                resolved_model_cli = args[11]
            if len(args) >= 13:
                termination_reason = args[12]
            if len(args) >= 14:
                binary_distance_abs = args[13]
            if len(args) >= 15:
                binary_distance_rel = args[14]
            if len(args) >= 16:
                largest_true = args[15]
            if len(args) >= 17:
                smallest_false = args[16]
            if len(args) >= 18:
                finished_flag = args[17]
        else:
            # Environment variables set in-process by experiment_automation.py
            experiment_type = os.environ.get('EXPERIMENT_TYPE', '')
            min_input_tokens = os.environ.get('MIN_INPUT_TOKENS', '')
            max_input_tokens = os.environ.get('MAX_INPUT_TOKENS', '')
            min_output_tokens = os.environ.get('MIN_OUTPUT_TOKENS', '')
            max_output_tokens = os.environ.get('MAX_OUTPUT_TOKENS', '')
            req_min = os.environ.get('REQ_MIN', '')
            evaluation_flag = os.environ.get('EVALUATION', '')
            resolved_model_cli = os.environ.get('MODEL_USED_RESOLVED', '')
            termination_reason = os.environ.get('TERMINATION_REASON', '')
            binary_distance_abs = os.environ.get('BINARY_SEARCH_DISTANCE', '')
            binary_distance_rel = os.environ.get('BINARY_SEARCH_RELATIVE_DISTANCE', '')
            largest_true = os.environ.get('LARGEST_TRUE', '')
            smallest_false = os.environ.get('SMALLEST_FALSE', '')
            finished_flag = os.environ.get('FINISHED', '')

        # Normalize finished flag to TRUE/FALSE if possible.
        if str(finished_flag).strip().lower() in {'true', '1', 'yes'}:
            finished_flag = 'TRUE'
        elif str(finished_flag).strip().lower() in {'false', '0', 'no'}:
            finished_flag = 'FALSE'

        # EXPERIMENT_TYPE travels through the environment, never argv (store_args has no slot
        # for it). experiment_automation.py exports the resolved value before spawning this
        # script; when it is invoked outside the automation, fall back to the .env files, as
        # done for DURATION/URL below, so the column is never written empty.
        experiment_type = str(experiment_type or "").strip().upper()
        if not experiment_type:
            for env_candidate in (Path(__file__).resolve().parent.parent / ".env", Path("..") / ".env"):
                experiment_type = _read_env_value(env_candidate, "EXPERIMENT_TYPE", "").strip().upper()
                if experiment_type:
                    break
        print(f"Experiment type: {experiment_type or '<unknown>'}")

        # Create the full directory path
        if parent_dir:
            full_dir_path = os.path.join(parent_dir, dir_name)
            os.makedirs(parent_dir, exist_ok=True)  # Ensure parent directory exists
        else:
            full_dir_path = dir_name
        
        os.makedirs(full_dir_path, exist_ok=True)
        print(f"Created directory: {full_dir_path}")

        # Fallback to .env only if still missing. Additive runs keep the four token-interval
        # columns empty on purpose (WORKLOAD_MIX names the experiment), so they are never
        # backfilled from .env here.
        if (min_input_tokens == '' or max_input_tokens == '' or min_output_tokens == '' or max_output_tokens == '' or req_min == '') and os.path.exists('../.env'):
            with open('../.env', 'r') as env_file:
                for line in env_file:
                    line = line.strip()
                    if not additive and min_input_tokens == '' and line.startswith('MIN_INPUT_TOKENS='):
                        min_input_tokens = line.split('=', 1)[1]
                    elif not additive and max_input_tokens == '' and line.startswith('MAX_INPUT_TOKENS='):
                        max_input_tokens = line.split('=', 1)[1]
                    elif not additive and min_output_tokens == '' and line.startswith('MIN_OUTPUT_TOKENS='):
                        min_output_tokens = line.split('=', 1)[1]
                    elif not additive and max_output_tokens == '' and line.startswith('MAX_OUTPUT_TOKENS='):
                        max_output_tokens = line.split('=', 1)[1]
                    elif req_min == '' and line.startswith('REQ_MIN='):
                        req_min = line.split('=', 1)[1]
                    elif evaluation_flag == '' and line.startswith('EVALUATION='):
                        evaluation_flag = line.split('=', 1)[1]

        if additive:
            # Additive rows do not belong to one token interval: the mix envelope passed in argv
            # exists only for the compact-layout detection, and the experiment is described by
            # WORKLOAD_MIX instead. Interval compliance is validated per profile against the
            # interval of the profile that served each request, so the envelope is blanked here
            # before anything else can use it.
            min_input_tokens = ''
            max_input_tokens = ''
            min_output_tokens = ''
            max_output_tokens = ''

        expected_min_output, expected_max_output = _parse_numeric_bounds(min_output_tokens, max_output_tokens)
        
        # Evaluation: use explicit flag from CLI/env; also attempt to read success rate from output.csv
        evaluation = (evaluation_flag or '').strip()
        # Normalize evaluation to TRUE/FALSE if possible
        if evaluation.lower() in {'true', '1', 'yes'}:
            evaluation = 'TRUE'
        elif evaluation.lower() in {'false', '0', 'no'}:
            evaluation = 'FALSE'

        success_rate = ''
        try:
            with open("output.csv", 'r', encoding="utf-8") as file:
                reader = csv.DictReader(file)
                row = next(reader, None)
                if row:
                    for k in row.keys():
                        if 'success' in k.lower():
                            success_rate = str(row.get(k) or '')
                            break
        except Exception:
            pass

        # Duration from .env
        duration = _read_env_value(Path('..') / '.env', 'DURATION', '')

        # Endpoint URL used for this run
        endpoint_candidates = [
            os.environ.get('URL', ''),
            os.environ.get('FMPERF_ENDPOINT_URL', ''),
            os.environ.get('ENDPOINT_URL', ''),
            _read_env_value(Path('..') / '.env', 'URL', ''),
            _read_env_value(Path('..') / '.env', 'FMPERF_ENDPOINT_URL', ''),
        ]

        # Some launch paths pass the endpoint in MODEL positional arg.
        if _is_endpoint_like(model):
            endpoint_candidates.append(model)
        url = ''
        for endpoint_value in endpoint_candidates:
            normalized = _normalize_endpoint_value(endpoint_value)
            if normalized:
                url = normalized
                break

        # Prompt token count: use median across prompts in requests
        prompt_token_count = _compute_median_prompt_tokens()

        # Job ID and Slurm model extraction
        job_id_env = os.environ.get('SLURM_JOB_ID')
        job_id, slurm_path = _find_slurm_log(job_id_env)

        # Median response tokens: prefer CLI-provided value from experiment_automation;
        # fall back to log-parsed value, then computed from results.json/output.csv
        median_resp_tokens = median_cli if (median_cli and str(median_cli).strip() != '') else None
        if median_resp_tokens is None:
            log_median = _extract_median_tokens_from_log(slurm_path)
            median_resp_tokens = log_median if log_median else None
        stats = _compute_request_token_stats(
            Path('results.json'),
            expected_min_output,
            expected_max_output,
            workload_profiles or None,
        )
        comp_median = stats.get("median_response_tokens")
        total_requests = stats.get("total_requests")
        sr_from_results = stats.get("success_rate")
        if median_resp_tokens is None:
            median_resp_tokens = comp_median
        if not success_rate and sr_from_results:
            success_rate = sr_from_results

        responses_within_interval = stats.get("responses_within_interval")
        responses_outside_interval = stats.get("responses_outside_interval")
        avg_tokens_per_request = stats.get("avg_tokens_per_request")
        avg_tokens_per_response = stats.get("avg_tokens_per_response")
        input_token_variance = stats.get("input_token_variance")
        output_token_variance = stats.get("output_token_variance")
        input_token_percentiles = stats.get("input_token_percentiles")
        output_token_percentiles = stats.get("output_token_percentiles")
        request_total_token_percentiles = stats.get("request_total_token_percentiles")
        additive_expected_proportions = stats.get("additive_expected_proportions") or ''
        additive_true_proportions = stats.get("additive_true_proportions") or ''
        if additive:
            # stdout is part of the interface: both proportions are printed so the achieved mix can
            # be checked from the Slurm log without opening results.csv.
            print(f"Additive expected proportions: {additive_expected_proportions or '{}'}")
            print(f"Additive true proportions: {additive_true_proportions or '{}'}")

        model_used = _resolve_model_used_fresh(
            resolved_model_cli=resolved_model_cli,
            model_arg=model,
            endpoint_candidates=endpoint_candidates,
            timeout_seconds=float(os.environ.get('MODEL_DISCOVERY_TIMEOUT', '10')),
        )

        # GPU_COUNT: number of GPUs used by the model-serving Slurm job (best-effort)
        gpu_count = _resolve_gpu_count(model_used, url)

        # Create new CSV file
        new_csv_path = os.path.join(full_dir_path, "results.csv")
        with open(new_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header with requested fields (remove GPUS/CPUS)
            writer.writerow([
                "EXPERIMENT_TYPE","MODEL_USED",
                "MIN_INPUT_TOKENS", "MAX_INPUT_TOKENS",
                "MIN_OUTPUT_TOKENS", "MAX_OUTPUT_TOKENS",
                "REQ_MIN", "EVALUATION",
                "DURATION", "URL", GPU_COUNT_FIELD, "TOTAL_REQUESTS", "SUCCESS_RATE", "MEDIAN_PROMPT_TOKENS",
                "MEDIAN_RESPONSE_TOKENS", "JOB_ID", "STAGE",
                "RESPONSES_WITHIN_EXPECTED_INTERVAL", "RESPONSES_OUTSIDE_EXPECTED_INTERVAL",
                "AVG_TOKENS_PER_REQUEST", "AVG_TOKENS_PER_RESPONSE",
                "INPUT_TOKEN_VARIANCE", "OUTPUT_TOKEN_VARIANCE",
                "INPUT_TOKEN_PERCENTILES", "OUTPUT_TOKEN_PERCENTILES", "REQUEST_TOTAL_TOKEN_PERCENTILES",
                "ADDITIVE_EXPECTED_PROPORTIONS", "ADDITIVE_TRUE_PROPORTIONS",
                "TERMINATION_REASON", "BINARY_SEARCH_DISTANCE", "BINARY_SEARCH_RELATIVE_DISTANCE",
                "LARGEST_TRUE", "SMALLEST_FALSE", "FINISHED",
                "WORKLOAD_MIX", "ADDITIVE"
            ])
            # Write data row
            writer.writerow([
                experiment_type,
                model_used,
                min_input_tokens,
                max_input_tokens,
                min_output_tokens,
                max_output_tokens,
                req_min,
                evaluation,
                duration,
                url,
                gpu_count or '',
                total_requests or '',
                success_rate or '',
                prompt_token_count or '',
                median_resp_tokens or '',
                job_id or '',
                stage,
                responses_within_interval or '',
                responses_outside_interval or '',
                avg_tokens_per_request or '',
                avg_tokens_per_response or '',
                input_token_variance or '',
                output_token_variance or '',
                input_token_percentiles or '',
                output_token_percentiles or '',
                request_total_token_percentiles or '',
                additive_expected_proportions or '',
                additive_true_proportions or '',
                termination_reason or '',
                binary_distance_abs or '',
                binary_distance_rel or '',
                largest_true or '',
                smallest_false or '',
                finished_flag or '',
                workload_mix or '',
                'TRUE' if additive else 'FALSE',
            ])
        
        print(f"Created results.csv in {full_dir_path}")
        
        # Create prompts.csv with unique prompts used
        _write_prompts_csv(full_dir_path)
        
        # Move CSV files to the new directory when available
        for csv_name in ("output.csv", "first_half.csv", "second_half.csv"):
            if os.path.exists(csv_name):
                shutil.move(csv_name, os.path.join(full_dir_path, csv_name))
            else:
                print(f"Warning: {csv_name} not found; skipping move.")
        # Keep a copy of results.json in the working directory for downstream readers
        shutil.copyfile("results.json", os.path.join(full_dir_path, "results.json"))
        
        print("Moved CSV files and copied results.json to the directory")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
