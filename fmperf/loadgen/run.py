import time
import copy
import requests
import sys
from typing import Iterable, List
import json
import pandas as pd
import os
from durations import Duration
import numpy as np
from fmperf.utils.approx import approx
import grpc
from google.protobuf import json_format
from fmperf.utils import parse_results
from datetime import datetime
from pathlib import Path
from .collect_energy import collect_metrics, summarize_energy
from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME, RESULTS_FILENAME, RESULTS_DIR
import threading
import itertools
import math


class ModelDiscoveryError(RuntimeError):
    """Raised when runtime model discovery from endpoint fails."""


def run(result_filename=None):
    if result_filename is None:
        result_filename = RESULTS_FILENAME

    def get_streaming_response_tgis(response, request_timeout, ttft_timeout, tpot_timeout):
        stop = False
        generated_tokens = 0
        request_start_time = time.time_ns()
        first_token_received = False
        
        while not stop:
            try:
                # Check request timeout
                current_time = time.time_ns()
                if (current_time - request_start_time) / 1e9 > request_timeout:
                    yield None, 0, current_time, False, TimeoutError(f"Request timeout: {request_timeout}s exceeded")
                    return
                
                x = next(response)
                timestamp = time.time_ns()
                data = json_format.MessageToDict(x)
                # skip first response (tokenizer output only)
                if "inputTokenCount" not in data:
                    n_tokens = data["generatedTokenCount"] - generated_tokens
                    generated_tokens = data["generatedTokenCount"]
                    
                    # Check TTFT timeout for first token
                    if not first_token_received:
                        ttft = (timestamp - request_start_time) / 1e9
                        if ttft > ttft_timeout:
                            yield None, 0, timestamp, False, TimeoutError(f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)")
                            return
                        first_token_received = True
                    
                    yield data, n_tokens, timestamp, True, None
            except Exception as e:
                timestamp = time.time_ns()
                yield None, 0, timestamp, False, e

    def get_streaming_response_vllm(response, request_timeout, ttft_timeout, tpot_timeout):
        response_iter = response.iter_lines(
            chunk_size=8192,
            decode_unicode=False,
            delimiter=b"\n",
        )

        stop = False
        prev_completion_tokens = 0
        request_start_time = time.time_ns()
        first_token_received = False
        last_token_time = request_start_time
        
        while not stop:
            try:
                # Check request timeout
                current_time = time.time_ns()
                if (current_time - request_start_time) / 1e9 > request_timeout:
                    yield None, 0, current_time, False, TimeoutError(f"Request timeout: {request_timeout}s exceeded")
                    return
                
                chunk = next(response_iter)
                timestamp = time.time_ns()
                
                # Check TPOT timeout for subsequent tokens
                if first_token_received and (timestamp - last_token_time) / 1e9 > tpot_timeout:
                    yield None, 0, timestamp, False, TimeoutError(f"TPOT timeout: {tpot_timeout}s exceeded")
                    return
                
                if chunk and not stop:
                    data = chunk.decode("utf-8").strip().split("data: ")[1]
                    out = json.loads(data)["choices"][0]
                    stop = out["finish_reason"] is not None
                    usage = json.loads(data)["usage"]
                    token_count = usage["completion_tokens"] - prev_completion_tokens
                    prev_completion_tokens = usage["completion_tokens"]
                    
                    # Check TTFT timeout for first token
                    if not first_token_received:
                        ttft = (timestamp - request_start_time) / 1e9
                        if ttft > ttft_timeout:
                            yield None, 0, timestamp, False, TimeoutError(f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)")
                            return
                        first_token_received = True
                    
                    for i in range(token_count):
                        yield {
                            "index": out["index"],
                            "text": "" if (i < token_count - 1) else out["text"],
                            "logprobs": None,
                            "finish_reason": (
                                None if (i < token_count - 1) else out["finish_reason"]
                            ),
                            "stop_reason": (
                                None if (i < token_count - 1) else out["stop_reason"]
                            ),
                        }, 1, timestamp, True, None
                    
                    last_token_time = timestamp
            except Exception as e:
                timestamp = time.time_ns()
                yield None, 0, timestamp, False, e

        # we have stopped
        yield None, 0, time.time_ns(), False, StopIteration()

    def _parse_int_env(key):
        value = os.environ.get(key)
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _discover_model_from_url(url, timeout_seconds):
        response = requests.get(
            "http://%s/v1/models" % (url),
            timeout=timeout_seconds,
        )
        if response.status_code != 200:
            raise ModelDiscoveryError(
                f"Model discovery failed with status {response.status_code}: {response.text}"
            )

        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list) or len(models) == 0:
            raise ModelDiscoveryError("Model discovery returned an empty model list")

        model_id = models[0].get("id") if isinstance(models[0], dict) else None
        if not model_id:
            raise ModelDiscoveryError("Model discovery did not return a valid model id")

        return str(model_id)

    def _get_output_token_override_bounds():
        min_value = _parse_int_env("MIN_OUTPUT_TOKENS")
        max_value = _parse_int_env("MAX_OUTPUT_TOKENS")
        if min_value is None and max_value is None:
            return None
        if min_value is None:
            min_value = max_value
        if max_value is None:
            max_value = min_value
        if min_value is None or max_value is None:
            return None
        if min_value <= 0 or max_value <= 0:
            raise ValueError("Output token bounds must be positive")
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        return (min_value, max_value)

    def _build_request_payload(template_request, target, rng, override_bounds, model_override):
        payload = copy.deepcopy(template_request)
        if model_override:
            if target == "vllm":
                payload["model"] = model_override
            elif target == "tgis":
                payload["model_id"] = model_override
        if not override_bounds:
            return payload, None
        min_tokens, max_tokens = override_bounds
        if max_tokens == min_tokens:
            desired_tokens = min_tokens
        else:
            desired_tokens = int(rng.randint(low=min_tokens, high=max_tokens + 1))
        if target == "vllm":
            payload["min_tokens"] = desired_tokens
            payload["max_tokens"] = desired_tokens
        elif target == "tgis":
            params = payload.setdefault("params", {})
            stopping = params.setdefault("stopping", {})
            stopping["minNewTokens"] = desired_tokens
            stopping["maxNewTokens"] = desired_tokens
        return payload, desired_tokens

    def _load_additive_mix():
        """Load the WORKLOAD_MIX_SPEC payload of an additive run (None when not additive).

        MoST additive runs (WORKLOAD_MIXES) export one profile per (token interval, alpha) pair,
        each one pointing at the requests file to sample prompts from. The payload is validated
        here, before any request is sent, so a malformed mix fails fast instead of wasting a whole
        iteration.
        """
        if os.environ.get("ADDITIVE", "").strip().upper() not in ("TRUE", "1", "YES"):
            return None
        raw = os.environ.get("WORKLOAD_MIX_SPEC", "").strip()
        if not raw:
            raise ValueError(
                "ADDITIVE is enabled but WORKLOAD_MIX_SPEC is empty; the workload mix "
                "specification is required to route every request to its profile."
            )
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"WORKLOAD_MIX_SPEC is not valid JSON: {exc}") from exc
        raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError("WORKLOAD_MIX_SPEC must contain a non-empty 'profiles' list")
        profiles = []
        for raw_profile in raw_profiles:
            filename = str(raw_profile.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"WORKLOAD_MIX_SPEC profile without a requests file: {raw_profile}")
            path = os.path.join(REQUESTS_DIR, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Additive profile workload not found: {path}")
            with open(path, "rb") as handle:
                cases = json.load(handle)
            if not isinstance(cases, list) or not cases:
                raise ValueError(f"Additive profile workload is empty: {path}")
            profile = {
                "label": str(raw_profile.get("label") or filename),
                "filename": filename,
                "alpha": float(raw_profile.get("alpha", 0.0)),
                "bounds": (
                    int(raw_profile.get("out_min", 0)),
                    int(raw_profile.get("out_max", 0)),
                ),
                "cases": cases,
            }
            profiles.append(profile)
        total_alpha = sum(profile["alpha"] for profile in profiles)
        if total_alpha <= 0:
            raise ValueError("WORKLOAD_MIX_SPEC profiles must have positive alphas")
        print(f">> Additive workload mix {payload.get('mix', '')}")
        for profile in profiles:
            print(
                f"   profile {profile['label']} alpha={profile['alpha'] / total_alpha:.6g} "
                f"output bounds {profile['bounds'][0]}-{profile['bounds'][1]} "
                f"({len(profile['cases'])} prompts from {profile['filename']})"
            )
        return {"mix": payload.get("mix", ""), "profiles": profiles}

    def _choose_profile(rs):
        """Pick a workload profile at random, weighted by its alpha.

        Additive runs only: the profiles (and their uses) live in `additive_profiles`, which is
        loaded right before the workers start.
        """
        total = sum(profile["alpha"] for profile in additive_profiles)
        draw = rs.uniform(0.0, total)
        accumulated = 0.0
        for profile in additive_profiles:
            accumulated += profile["alpha"]
            if draw <= accumulated:
                return profile
        return additive_profiles[-1]

    def _with_profile(record, profile_label):
        """Tag an event with the workload profile that served it (additive runs only).

        Non-additive runs keep the historical results.json event schema untouched.
        """
        if profile_label:
            record["workload_profile"] = profile_label
        return record

    output_token_override = _get_output_token_override_bounds()

    infile = os.path.join(REQUESTS_DIR, REQUESTS_FILENAME)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, result_filename)
    target = os.environ.get("TARGET", "vllm")
    api_url = os.environ["URL"]
    model_discovery_timeout = float(os.environ.get("MODEL_DISCOVERY_TIMEOUT", "10"))
    service_type = os.environ.get('SERVICE_TYPE', 'LLM')

    active_model = None
    if service_type != 'SaaS':
        # Discover the model directly from the endpoint before starting the experiment.
        # If discovery fails, abort early as requested.
        try:
            active_model = _discover_model_from_url(api_url, model_discovery_timeout)
        except Exception as exc:
            raise ModelDiscoveryError(
                f"Unable to discover model from URL '{api_url}'. Aborting experiment early."
            ) from exc
        print(f">> Discovered model from endpoint: {active_model}")
    else:
        active_model = 'SaaS'
        print(">> Running in SaaS Mode")

    active_use_case = None
    yaml_path = os.environ.get('USE_CASES_YAML')
    if service_type == 'SaaS' and yaml_path:
        import yaml
        try:
            # normalize/preprocess yaml colons
            with open(yaml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = []
            for l in content.splitlines():
                if ':' in l:
                    parts = l.split(':', 1)
                    if not parts[1].startswith(' '):
                        l = f"{parts[0]}: {parts[1]}"
                lines.append(l)
            data = yaml.safe_load('\n'.join(lines))
            use_cases = []
            if isinstance(data, dict):
                if 'useCase' in data:
                    use_cases = [data['useCase']]
                elif 'useCases' in data:
                    val = data['useCases']
                    use_cases = val if isinstance(val, list) else [val]
                else:
                    use_cases = [data]
            elif isinstance(data, list):
                use_cases = data
            
            active_id = os.environ.get('ACTIVE_USE_CASE_ID')
            for uc in use_cases:
                if uc.get('id') == active_id:
                    active_use_case = uc
                    break
            print(f">> Loaded SaaS active use case: {active_id}")
        except Exception as e:
            print(f"Error loading use case in loadgen: {e}")
    req_min = float(os.environ["REQ_MIN"])  # Changed from int() to float() to allow non-integer values
    duration = Duration(os.environ["DURATION"])
    backoff = Duration(os.environ["BACKOFF"])
    grace_period = Duration(os.environ["GRACE_PERIOD"])
    
    # Get timeout values from environment variables with defaults
    request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))  # 5 minutes default
    ttft_timeout = float(os.environ.get("TTFT_TIMEOUT", "60"))  # 60 seconds default
    tpot_timeout = float(os.environ.get("TPOT_TIMEOUT", "30"))  # 30 seconds default

    additive_mix = _load_additive_mix()
    additive_profiles = additive_mix["profiles"] if additive_mix else None
    if additive_profiles:
        # Additive run: every profile owns its requests file, so the mix-envelope file named by
        # REQUESTS_FILENAME is neither read nor required here.
        print(
            ">> Additive runs tag every event with 'workload_profile' and draw the output length "
            "from the profile interval; the 'consistent' flag stays informational because the "
            "requests files are generated with the union of the mix output intervals."
        )
        sample_requests = []
    else:
        with open(infile, "rb") as f:
            sample_requests = json.load(f)

    progress_lock = threading.Lock()
    scheduled_by_worker = {}
    inflight_by_worker = {}

    def worker(wid, channel, worker_req_per_sec, exp_num_users):
        rs = np.random.RandomState(seed=wid)
        rs_lock = threading.Lock()
        stub = None
        if target == "tgis":
            from text_generation_tests.pb import generation_pb2_grpc as gpb2

            stub = gpb2.GenerationServiceStub(channel)
        
        # Calculate requests per second for this worker with some randomness
        # worker_req_per_sec is the target per worker (REQ_MIN split by num_workers)
        variation = rs.uniform(0.8, 1.2)
        worker_req_per_sec = worker_req_per_sec * variation
        
        # Calculate interval between requests for this worker
        if worker_req_per_sec > 0:
            base_interval = 1.0 / worker_req_per_sec
            jitter_range = 0.3  # ±30% jitter
        else:
            base_interval = float('inf')
            jitter_range = 0.0

        t_start = time.time_ns()

        if service_type == 'SaaS':
            # Run SaaS pacing worker logic
            output = []
            request_counter = itertools.count()
            
            # Pacing based on worker_req_per_sec (use-case iterations per second)
            next_iteration_time = t_start
            interval_ns = (1.0 / worker_req_per_sec) * 1e9 if worker_req_per_sec > 0 else float('inf')
            
            endpoints = active_use_case.get('endpoints', {}) if active_use_case else {}
            endpoint_list = list(endpoints.items())
            
            while (time.time_ns() - t_start) < duration.to_seconds() * 1e9:
                current_time = time.time_ns()
                if current_time < next_iteration_time:
                    time.sleep((next_iteration_time - current_time) / 1e9)
                
                req_idx = next(request_counter)
                response_idx = 0
                
                for ep_name, ep_config in endpoint_list:
                    t0 = time.time_ns()
                    method = ep_config.get('http', 'GET').upper()
                    path = ep_config.get('url', '')
                    
                    # Resolve path params
                    pathparams = ep_config.get('pathparams', {}) or {}
                    for param_name, param_type in pathparams.items():
                        val = "dummy_str"
                        if str(param_type).lower() in ('integer', 'int'):
                            val = "1"
                        path = path.replace(f"{{{param_name}}}", val).replace(f":{param_name}", val)
                    
                    # Resolve query params
                    queryparams = ep_config.get('queryparams', {}) or {}
                    resolved_query = {}
                    for param_name, param_type in queryparams.items():
                        val = "dummy_str"
                        if str(param_type).lower() in ('integer', 'int'):
                            val = 1
                        resolved_query[param_name] = val
                    
                    # Resolve body
                    body_file = ep_config.get('body')
                    json_body = None
                    if body_file:
                        body_path = Path(body_file)
                        if not body_path.is_absolute():
                            yaml_dir = Path(yaml_path).parent if yaml_path else Path('.')
                            body_path = (yaml_dir / body_path).resolve()
                        if body_path.is_file():
                            try:
                                with open(body_path, 'r', encoding='utf-8') as bf:
                                    json_body = json.load(bf)
                            except Exception:
                                pass
                        if json_body is None:
                            json_body = {}
                    
                    # Construct URL
                    endpoint_base_url = os.environ["URL"]
                    if not endpoint_base_url.startswith("http://") and not endpoint_base_url.startswith("https://"):
                        endpoint_base_url = f"http://{endpoint_base_url}"
                    
                    full_url = f"{endpoint_base_url.rstrip('/')}/{path.lstrip('/')}"
                    
                    ok = False
                    error_msg = "None"
                    resp_json = None
                    try:
                        headers = {"User-Agent": "most-load-test"}
                        resp = requests.request(
                            method=method,
                            url=full_url,
                            params=resolved_query,
                            json=json_body,
                            headers=headers,
                            timeout=request_timeout
                        )
                        ok = (200 <= resp.status_code < 300)
                        if not ok:
                            error_msg = f"HTTP {resp.status_code}"
                        try:
                            resp_json = resp.json()
                        except Exception:
                            resp_json = resp.text[:200]
                    except Exception as e:
                        ok = False
                        error_msg = str(e)
                    
                    t = time.time_ns()
                    record = {
                        "response": resp_json,
                        "ok": ok,
                        "error": error_msg,
                        "timestamp": t,
                        "exp_req_min": req_min,
                        "exp_duration": duration.to_seconds(),
                        "duration_ms": (t - t0) / 1000.0 / 1000.0,
                        "exclude": (t - t_start) / 1e9 > (duration.to_seconds() + grace_period.to_seconds()),
                        "worker_idx": wid,
                        "request_idx": req_idx,
                        "sample_idx": 0,
                        "response_idx": response_idx,
                        "n_tokens": 0,
                        "exp_num_users": exp_num_users,
                        "endpoint": ep_name,
                        "url": full_url
                    }
                    output.append(record)
                    response_idx += 1
                
                next_iteration_time += int(interval_ns)
            
            with open("results_wid%d" % (wid), "w") as f:
                json.dump(output, f)
            return True

        output = []
        output_lock = threading.Lock()
        request_counter = itertools.count()
        requests_scheduled = 0

        # Track in-flight requests for cleanup; agnostic to expected duration
        inflight = set()  # set[threading.Thread]

        # progress logging: print remaining every LOG_INTERVAL seconds
        LOG_INTERVAL = 5.0
        last_log_time = t_start
        
        # Driftless scheduler: schedule first request immediately and then at fixed intervals with jitter
        # This avoids skipping the first interval window which caused under-sending at higher RPMs.
        next_request_time = t_start  # first request goes out immediately
        if worker_req_per_sec > 0:
            interval_base_ns = (1.0 / worker_req_per_sec) * 1e9
        else:
            interval_base_ns = float('inf')

        def process_request(req_idx):
            # Pick a sample request (thread-safe selection). Additive runs choose the workload
            # profile first (weighted by alpha), then a random prompt inside that profile's own
            # requests file, and draw the output length from the profile interval instead of the
            # global MIN/MAX_OUTPUT_TOKENS override.
            profile_label = None
            if additive_profiles:
                with rs_lock:
                    profile = _choose_profile(rs)
                    sample_idx = rs.randint(low=0, high=len(profile["cases"]))
                template_request = profile["cases"][sample_idx]["request"]
                request_payload, _ = _build_request_payload(
                    template_request, target, rs, profile["bounds"], active_model
                )
                profile_label = profile["label"]
            else:
                with rs_lock:
                    sample_idx = rs.randint(low=0, high=len(sample_requests))
                template_request = sample_requests[sample_idx]["request"]
                request_payload, _ = _build_request_payload(
                    template_request, target, rs, output_token_override, active_model
                )

            if target == "vllm":
                headers = {"User-Agent": "fmaas-load-test"}
                t0 = time.time_ns()
                try:
                    response = requests.post(
                        "http://%s/v1/completions" % (api_url),
                        headers=headers,
                        json=request_payload,
                        stream=True,
                        timeout=request_timeout
                    )
                except requests.exceptions.Timeout:
                    timestamp = time.time_ns()
                    record = {
                        "response": None,
                        "ok": False,
                        "error": f"Request timeout: {request_timeout}s exceeded",
                        "timestamp": timestamp,
                        "exp_req_min": req_min,
                        "exp_duration": duration.to_seconds(),
                        "duration_ms": (timestamp - t0) / 1000.0 / 1000.0,
                        "exclude": (timestamp - t_start) / 1000.0 / 1000.0 / 1000.0
                        > (duration.to_seconds() + grace_period.to_seconds()),
                        "worker_idx": wid,
                        "request_idx": req_idx,
                        "sample_idx": sample_idx,
                        "response_idx": 0,
                        "n_tokens": 0,
                        "exp_num_users": exp_num_users,
                    }
                    with output_lock:
                        output.append(_with_profile(record, profile_label))
                    time.sleep(backoff.to_seconds())
                    return True
            elif target == "tgis":
                from text_generation_tests.pb import generation_pb2 as pb2
                message = json_format.ParseDict(request_payload, pb2.SingleGenerationRequest())
                t0 = time.time_ns()
                response = stub.GenerateStream(message)
            else:
                raise ValueError(f"Invalid target: {target}")

            stop = False
            response_idx = 0

            if target == "vllm":
                response_generator = get_streaming_response_vllm(response, request_timeout, ttft_timeout, tpot_timeout)
            elif target == "tgis":
                response_generator = get_streaming_response_tgis(response, request_timeout, ttft_timeout, tpot_timeout)
            else:
                raise ValueError(f"Invalid target: {target}")

            apply_backoff = False

            while not stop:
                r, n_tokens, t, ok, err = next(response_generator)

                if not ok:
                    stop = True
                    # check if we have reached end of stream
                    if type(err) is StopIteration:
                        continue
                    else:
                        apply_backoff = True

                record = {
                    "response": r,
                    "ok": ok,
                    "error": str(err),
                    "timestamp": t,
                    "exp_req_min": req_min,
                    "exp_duration": duration.to_seconds(),
                    "duration_ms": (t - t0) / 1000.0 / 1000.0,
                    "exclude": (t - t_start) / 1000.0 / 1000.0 / 1000.0
                    > (duration.to_seconds() + grace_period.to_seconds()),
                    "worker_idx": wid,
                    "request_idx": req_idx,
                    "sample_idx": sample_idx,
                    "response_idx": response_idx,
                    "n_tokens": n_tokens,
                    "exp_num_users": exp_num_users,
                }

                with output_lock:
                    output.append(_with_profile(record, profile_label))
                response_idx += 1
                t0 = t

            if apply_backoff:
                time.sleep(backoff.to_seconds())

            return True

        # Scheduler loop: use next_request_time to avoid drift and ensure the first request is immediate
        # Only schedule when we have a positive target rate
        while worker_req_per_sec > 0 and (next_request_time - t_start) < duration.to_seconds() * 1e9:
            current_time = time.time_ns()
            if current_time < next_request_time:
                time.sleep((next_request_time - current_time) / 1e9)

            # Prune finished threads
            finished = [t for t in inflight if not t.is_alive()]
            for t in finished:
                inflight.discard(t)
            with progress_lock:
                inflight_by_worker[wid] = len(inflight)

            # Schedule a new request by starting a dedicated thread
            req_idx = next(request_counter)
            th = threading.Thread(target=process_request, args=(req_idx,), daemon=True)
            th.start()
            inflight.add(th)
            requests_scheduled += 1
            with progress_lock:
                scheduled_by_worker[wid] = requests_scheduled
                inflight_by_worker[wid] = len(inflight)
            
            # Compute next schedule time with jitter around the base interval
            jitter = rs.uniform(1 - jitter_range, 1 + jitter_range)
            next_request_time += int(interval_base_ns * jitter)

            # progress logging: print remaining time every LOG_INTERVAL seconds
            now_ns = time.time_ns()
            elapsed_s = (now_ns - t_start) / 1e9
            remaining_s = duration.to_seconds() - elapsed_s
            if remaining_s < 0:
                remaining_s = 0.0
            if (now_ns - last_log_time) / 1e9 >= LOG_INTERVAL:
                with progress_lock:
                    total_scheduled = sum(scheduled_by_worker.values())
                    total_inflight = sum(inflight_by_worker.values())
                elapsed_minutes = elapsed_s / 60.0 if elapsed_s > 0 else 0.0
                avg_sent_rpm = (total_scheduled / elapsed_minutes) if elapsed_minutes > 0 else 0.0
                print(
                    f"[worker {wid}] remaining: {remaining_s:.1f}s "
                    f"(elapsed: {elapsed_s:.1f}s, total reqs scheduled: {total_scheduled}, total inflight: {total_inflight}, avg sent rpm: {avg_sent_rpm:.2f})"
                )
                last_log_time = now_ns

        # Wait for all in-flight request threads to finish
        for th in list(inflight):
            th.join()
        with progress_lock:
            inflight_by_worker[wid] = 0

        with open("results_wid%d" % (wid), "w") as f:
            json.dump(output, f)

        return True

    from datetime import datetime
    import concurrent.futures

    energy_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    channel = grpc.insecure_channel(api_url) if target == "tgis" else None

    # Determine number of workers dynamically based on target RPM and per-worker capacity
    per_worker_rpm_capacity = int(os.environ.get("WORKER_RPM_CAPACITY", "15"))
    max_workers = int(os.environ.get("MAX_WORKERS", "64"))
    num_workers = max(1, min(max_workers, int(math.ceil(req_min / max(1, per_worker_rpm_capacity)))))
    print(f">> Using {num_workers} workers (capacity ~{per_worker_rpm_capacity} rpm/worker) to achieve {req_min} requests per minute")
    # Split the global REQ_MIN target across workers
    per_worker_req_per_sec = (req_min / max(1, num_workers)) / 60.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            futures.append(
                executor.submit(
                    worker,
                    wid=i,
                    channel=channel,
                    worker_req_per_sec=per_worker_req_per_sec,
                    exp_num_users=num_workers,
                )
            )

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    energy_stop_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    all_outputs = []
    for i in range(num_workers):
        with open("results_wid%d" % (i), "rb") as f:
            tmp = json.load(f)
        all_outputs.extend(tmp)

    def _case_for_row(row):
        """Resolve the request case that produced a results row (None when unresolvable).

        Additive events carry their profile label and their sample_idx indexes that profile's own
        requests file; non-additive events keep indexing the single workload file. Resolving by
        profile avoids matching a row against a prompt of another profile (or crashing on an
        out-of-range index when the profile file is smaller than the mix-envelope one).
        """
        index = row.get("sample_idx")
        if not isinstance(index, int):
            return None
        if additive_profiles:
            label = row.get("workload_profile")
            for profile in additive_profiles:
                if profile["label"] == label:
                    cases = profile["cases"]
                    return cases[index] if 0 <= index < len(cases) else None
            return None
        return sample_requests[index] if 0 <= index < len(sample_requests) else None

    def check_consistent(row):
        if not row["ok"]:
            return False
        case = _case_for_row(row)
        if case is None:
            return False

        # A workload file can be model-agnostic (or generated for another model).
        # Skip strict expected-output checks when expected data comes from a different model.
        expected_model = case.get("expected_model")
        req = case.get("request", {}) if isinstance(case, dict) else {}
        request_model = req.get("model") if isinstance(req, dict) else None
        request_model_id = req.get("model_id") if isinstance(req, dict) else None
        if active_model:
            if expected_model and expected_model != active_model:
                return True
            if request_model and request_model not in {"__ENV_MODEL__", active_model}:
                return True
            if request_model_id and request_model_id not in {"__ENV_MODEL__", "null", active_model}:
                return True

        expected = case.get("expected")
        if not expected:
            return False
        if row["response_idx"] >= len(expected):
            return False
        tmp = expected[row["response_idx"]]
        consistent = row["response"] == approx(tmp)
        return consistent

    for row in all_outputs:
        row["consistent"] = check_consistent(row)

    # collect and summarize energy metrics
    energy = {}
    if os.environ.get("PROM_URL") is None:
        print(
            ">> skipped collecting energy metrics because prometheus is not available."
        )
    else:
        step = os.environ.get("NUM_PROM_STEPS", "30")
        ns = os.environ["NAMESPACE"]
        collect_metrics(energy_start_time, energy_stop_time, step, ns)
        all_energy_metrics = summarize_energy(energy_start_time)
        print(all_energy_metrics)
        energy = all_energy_metrics[["num_users", "energy"]].to_dict()

    merged_data = {"results": all_outputs, "energy": energy}

    print(">> writing results to file: %s" % (outfile))
    with open(outfile, "w") as f:
        json.dump(merged_data, f)

    return all_outputs


if __name__ == "__main__":
    try:
        parse_results(run(), print_df=True)
    except ModelDiscoveryError as exc:
        print(f"Controlled stop: {exc}")
        sys.exit(2)
