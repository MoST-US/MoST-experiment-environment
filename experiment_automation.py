import csv
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME, RESULTS_FILENAME, RESULTS_DIR
from workload_mix import mix_spec_payload, output_bounds_for_input, parse_workload_mixes

REQUESTS_PROMPTS_FILE = Path("oasst_roots_en_max1000_tokens.jsonl")

# Result folders (per token-combo parent dirs) created by the current execution.
# They are moved into results/<EXPERIMENT_TYPE>_<timestamp> when the automation
# finishes (see _archive_execution_results).
CREATED_RESULT_DIRS: list[str] = []

# True while an additive (WORKLOAD_MIXES) execution is running: the archive folder of such a run
# is named Experiment_MIX_<EXPERIMENT_TYPE>_<timestamp> instead of Experiment_<EXPERIMENT_TYPE>_...
ADDITIVE_RUN_ACTIVE = False

#
# Configuration loader: read values from .env without modifying the file.
#
def _parse_tokens_list(value):
    """Parse TOKENS_LIST into input/output intervals.

    Supported formats (comma-separated):
    - "in:out"                -> [[in_min,in_max,out_min,out_max]] with in_min=in_max=in, out_min=out_max=out
    - "inMin-inMax:out"       -> [[inMin,inMax,out,out]]
    - "in:outMin-outMax"      -> [[in,in,outMin,outMax]]
    - "inMin-inMax:outMin-outMax" -> [[inMin,inMax,outMin,outMax]]

    Malformed entries are skipped. Whitespace is ignored.
    """
    tokens = []
    if not value:
        return tokens
    for item in value.split(','):
        s = item.strip()
        if not s:
            continue
        if ':' not in s:
            continue
        in_part, out_part = s.split(':', 1)
        in_part = in_part.strip()
        out_part = out_part.strip()
        try:
            if '-' in in_part:
                in_min_str, in_max_str = in_part.split('-', 1)
                in_min = int(in_min_str.strip())
                in_max = int(in_max_str.strip())
            else:
                in_min = in_max = int(in_part)
            if '-' in out_part:
                out_min_str, out_max_str = out_part.split('-', 1)
                out_min = int(out_min_str.strip())
                out_max = int(out_max_str.strip())
            else:
                out_min = out_max = int(out_part)
            tokens.append([in_min, in_max, out_min, out_max])
        except ValueError:
            continue
    return tokens

def _parse_int_list(value):
    """Parse comma-separated integers into a list. Invalid entries are skipped.

    Examples:
    - "1,2,4" -> [1, 2, 4]
    - "8"     -> [8]
    - "1, x"  -> [1]
    """
    items = []
    if value is None:
        return items
    for part in str(value).split(','):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p))
        except ValueError:
            # skip malformed entries
            continue
    return items

def load_use_cases_from_yaml(yaml_path):
    import yaml
    if not yaml_path:
        return []
    path = Path(yaml_path)
    if not path.is_absolute():
        path = (Path(__file__).parent / path).resolve()
    if not path.exists():
        print(f"Warning: Use cases YAML file not found: {path}")
        return []
    try:
        with path.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Preprocess colons without space (e.g. key:value -> key: value)
        lines = []
        for l in content.splitlines():
            if ':' in l:
                parts = l.split(':', 1)
                if not parts[1].startswith(' '):
                    l = f"{parts[0]}: {parts[1]}"
            lines.append(l)
        normalized = '\n'.join(lines)
        data = yaml.safe_load(normalized)
        
        # Normalize to list of dicts
        use_cases = []
        if isinstance(data, dict):
            if 'useCase' in data:
                use_cases = [data['useCase']]
            elif 'useCases' in data:
                val = data['useCases']
                if isinstance(val, list):
                    use_cases = val
                else:
                    use_cases = [val]
            else:
                use_cases = [data]
        elif isinstance(data, list):
            use_cases = data
        return use_cases
    except Exception as e:
        print(f"Error loading use cases from YAML: {e}")
        return []

def load_env_config():
    """Load configuration from .env file and return a dict.

    Expected keys:
    - TOKENS_LIST: comma-separated pairs like "32:32,32:64"
    - WORKLOAD_MIXES: additive experiments, e.g. "[(1-100:1-100,0.5),(300-600:100-300,0.5)]"
      (see workload_mix.py); it replaces TOKENS_LIST for additive runs
    - REQ_MIN_START: comma-separated integers (per-token-combo initial REQ_MIN)
    - REQ_MIN_INCREASE_MULTIPLIER: integer (multiplier for stage 1 success)
        - STOP_THRESHOLD: float (relative stop threshold used as
            M - m <= M * STOP_THRESHOLD in stage 2)
    """
    env_path = Path('.env')
    config = {
        'TOKENS_LIST': [],
        'REQ_MIN_START': [1],
        'REQ_MIN_INCREASE_MULTIPLIER': 2.0,
        'STOP_THRESHOLD': 0.5,
        'EXPERIMENT_TYPE': 'MST',
        'DURATION': None,
        'ITERATION_COOLDOWN_SECONDS': 180.0,
        'ITERATION_HARD_LIMIT': 15,
        'SERVICE_TYPE': 'LLM',
        'USE_CASES_YAML': None,
        'WORKLOAD_MIXES': [],
    }

    if env_path.exists():
        with env_path.open('r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#') or '=' not in s:
                    continue
                key, val = s.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key == 'TOKENS_LIST':
                    config['TOKENS_LIST'] = _parse_tokens_list(val)
                elif key == 'WORKLOAD_MIXES':
                    config['WORKLOAD_MIXES'] = parse_workload_mixes(val)
                elif key == 'SERVICE_TYPE':
                    config['SERVICE_TYPE'] = val.strip()
                elif key == 'USE_CASES_YAML':
                    config['USE_CASES_YAML'] = val.strip()
                elif key == 'REQ_MIN_START':
                    lst = _parse_int_list(val)
                    # Backwards compatibility: if parsing produced empty but val is a single int, wrap it
                    if not lst:
                        try:
                            lst = [int(val)]
                        except ValueError:
                            lst = config['REQ_MIN_START']
                    config['REQ_MIN_START'] = lst
                elif key == 'REQ_MIN_INCREASE_MULTIPLIER':
                    try:
                        config['REQ_MIN_INCREASE_MULTIPLIER'] = float(val)
                    except ValueError:
                        pass
                elif key == 'STOP_THRESHOLD':
                    try:
                        config['STOP_THRESHOLD'] = float(val)
                    except ValueError:
                        pass
                elif key == 'EXPERIMENT_TYPE':
                    config['EXPERIMENT_TYPE'] = val.strip() or 'MST'
                elif key == 'DURATION':
                    config['DURATION'] = val.strip()
                elif key == 'ITERATION_COOLDOWN_SECONDS':
                    try:
                        parsed = float(val)
                        if parsed >= 0:
                            config['ITERATION_COOLDOWN_SECONDS'] = parsed
                    except ValueError:
                        pass
                elif key == 'ITERATION_HARD_LIMIT':
                    try:
                        parsed_limit = int(val)
                        if parsed_limit > 0:
                            config['ITERATION_HARD_LIMIT'] = parsed_limit
                    except ValueError:
                        pass

    return config

CONFIG = load_env_config()

_DURATION_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhdSMHD]?)$")
MIT_PLATEAU_REL_TOL = float(os.environ.get('MIT_PLATEAU_REL_TOL', '0.01'))
MIT_PLATEAU_ABS_TOL = float(os.environ.get('MIT_PLATEAU_ABS_TOL', '0.5'))
SUCCESS_RATE_THRESHOLD = float(os.environ.get('SUCCESS_RATE_THRESHOLD', '95.0'))


def _read_env_value(env_path, key, default=''):
    try:
        path = Path(env_path)
        if not path.exists():
            return default
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                s = line.strip()
                if not s or s.startswith('#') or '=' not in s:
                    continue
                if s.startswith('export '):
                    s = s[len('export '):].strip()
                k, v = s.split('=', 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return default


def _normalize_endpoint_value(value):
    if value is None:
        return ''
    return str(value).strip().strip('"').strip("'")


def _model_hint_from_endpoint_value(value):
    endpoint = _normalize_endpoint_value(value)
    if not endpoint:
        return ''
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        path = (parsed.path or '').strip('/')
        if path:
            parts = [p for p in path.split('/') if p and p not in ('v1', 'chat', 'completions', 'generate', 'models')]
            if parts:
                return parts[-1]
        host = parsed.netloc.split(':', 1)[0].strip()
        return host
    return endpoint


def _extract_model_from_payload(payload):
    if isinstance(payload, dict):
        for key in ('model', 'model_name', 'name', 'id'):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        data = payload.get('data')
        if isinstance(data, list):
            for item in data:
                model = _extract_model_from_payload(item)
                if model:
                    return model

        models = payload.get('models')
        if isinstance(models, list):
            for item in models:
                model = _extract_model_from_payload(item)
                if model:
                    return model

    if isinstance(payload, list):
        for item in payload:
            model = _extract_model_from_payload(item)
            if model:
                return model

    return ''


def _build_model_probe_urls(url):
    if not url:
        return []
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return []

    original = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    path = parsed.path.rstrip('/')

    candidate_paths = [path]
    if path.endswith('/chat/completions'):
        candidate_paths.append(path[:-len('/chat/completions')] + '/models')
    if path.endswith('/completions'):
        candidate_paths.append(path[:-len('/completions')] + '/models')
    if path.endswith('/generate'):
        candidate_paths.append(path[:-len('/generate')] + '/info')

    candidate_paths.extend(['/v1/models', '/models', '/info'])

    seen = set()
    urls = []
    for p in candidate_paths:
        normalized = p if p.startswith('/') else '/' + p
        full = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, normalized, '', '', ''))
        if full not in seen:
            seen.add(full)
            urls.append(full)

    if original and original not in seen:
        urls.insert(0, original)
    return urls


def _extract_model_from_url(url, timeout_seconds=5.0):
    endpoint = _normalize_endpoint_value(url)
    if not endpoint:
        return ''
    # If this is not a full URL (e.g., service/model name), use it directly as model hint.
    if not urllib.parse.urlparse(endpoint).scheme:
        return _model_hint_from_endpoint_value(endpoint)

    for probe_url in _build_model_probe_urls(endpoint):
        try:
            req = urllib.request.Request(
                probe_url,
                headers={'Accept': 'application/json', 'User-Agent': 'experiment-automation/1.0'},
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read()
            text = raw.decode('utf-8', errors='replace')
            payload = json.loads(text)
            model = _extract_model_from_payload(payload)
            if model:
                return model
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
        except Exception:
            continue
    return ''


def _normalize_experiment_type(value):
    if not value:
        return 'MST'
    return value.strip().upper()


def get_experiment_type():
    """Return the experiment type from env or .env config."""
    env_val = os.environ.get('EXPERIMENT_TYPE')
    if env_val:
        return _normalize_experiment_type(env_val)
    return _normalize_experiment_type(CONFIG.get('EXPERIMENT_TYPE', 'MST'))


def _parse_duration_seconds(value):
    """Parse duration strings like '1800s', '30m', '2h' into seconds."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    match = _DURATION_PATTERN.match(s)
    if not match:
        try:
            return float(s)
        except ValueError:
            return None
    number = float(match.group('value'))
    unit = match.group('unit').lower()
    if unit == 'm':
        number *= 60
    elif unit == 'h':
        number *= 3600
    elif unit == 'd':
        number *= 86400
    return number


def _get_duration_seconds():
    """Resolve the experiment duration in seconds from env or config."""
    env_val = os.environ.get('DURATION')
    seconds = _parse_duration_seconds(env_val)
    if seconds is not None:
        return seconds
    return _parse_duration_seconds(CONFIG.get('DURATION'))


def _get_iteration_cooldown_seconds():
    """Resolve cooldown seconds between iterations from env or config."""
    env_val = os.environ.get('ITERATION_COOLDOWN_SECONDS')
    if env_val is not None:
        parsed_env = _parse_duration_seconds(env_val)
        if parsed_env is None:
            try:
                parsed_env = float(env_val)
            except ValueError:
                parsed_env = None
        if parsed_env is not None and parsed_env >= 0:
            return parsed_env

    config_val = CONFIG.get('ITERATION_COOLDOWN_SECONDS', 180.0)
    parsed_config = _parse_duration_seconds(config_val)
    if parsed_config is None:
        try:
            parsed_config = float(config_val)
        except (TypeError, ValueError):
            parsed_config = 180.0
    return parsed_config if parsed_config >= 0 else 180.0


def _get_iteration_hard_limit():
    """Resolve hard limit (integer iterations) for each token experiment."""
    env_val = os.environ.get('ITERATION_HARD_LIMIT')
    if env_val is not None:
        try:
            parsed_env = int(str(env_val).strip())
            if parsed_env > 0:
                return parsed_env
        except ValueError:
            pass

    config_val = CONFIG.get('ITERATION_HARD_LIMIT', 15)
    try:
        parsed_config = int(config_val)
        if parsed_config > 0:
            return parsed_config
    except (TypeError, ValueError):
        pass
    return 15


def _format_bound_number(value):
    """Format a confirmed REQ_MIN bound for results.csv.

    Returns '' when the bound is unknown (None), an integer-looking string for
    integral values, and a trimmed decimal otherwise.
    """
    if value is None:
        return ''
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip('0').rstrip('.')


def _store_args_contract_ok(store_positional: list[str]) -> bool:
    """Check the positional tail of the store_results.py argv against its compact layout.

    store_results.py infers the layout from these positions, so any argument inserted before
    model/stage/parent_dir shifts every results.csv column (parent_dir -> out_range,
    LARGEST_TRUE -> finished flag, SMALLEST_FALSE never written). The expected tail is
    (model, stage, parent_dir, in_range, out_range, req_min, evaluation, ...).
    """
    if len(store_positional) < 7:
        return False
    if str(store_positional[1]).strip() not in ('1', '2'):
        return False
    parent_dir_arg = str(store_positional[2]).strip()
    if not parent_dir_arg:
        return False
    # LLM mode always names the parent folder '<in_min>-<in_max>_<out_min>-<out_max>';
    # SaaS mode uses the use-case id, which may contain no underscore.
    if os.environ.get('SERVICE_TYPE') != 'SaaS' and '_' not in parent_dir_arg:
        return False
    return True


def _compute_success_rate_from_results():
    """Compute overall success rate (%) directly from results payload."""
    results_path = Path(RESULTS_DIR) / RESULTS_FILENAME
    if not results_path.exists():
        return None
    try:
        with results_path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"Warning: unable to read {results_path} for success rate check: {exc}")
        return None

    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    if not isinstance(rows, list) or not rows:
        return None

    request_groups = defaultdict(list)
    for item in rows:
        request_idx = item.get('request_idx')
        if request_idx is None:
            continue
        try:
            req_id = int(request_idx)
        except (TypeError, ValueError):
            continue
        worker_idx = item.get('worker_idx')
        key = (worker_idx if worker_idx is not None else -1, req_id)
        request_groups[key].append(item)

    total_requests = len(request_groups)
    if total_requests == 0:
        return None

    successful_requests = 0
    for group in request_groups.values():
        if all(entry.get('ok') and entry.get('error') == "None" for entry in group):
            successful_requests += 1

    return (successful_requests / total_requests) * 100.0


def _check_success_rate_threshold(threshold=None, prefer_results_json=False):
    """Check that success rates stay above threshold via CSVs or raw results."""
    effective_threshold = SUCCESS_RATE_THRESHOLD if threshold is None else threshold

    if prefer_results_json:
        success_rate = _compute_success_rate_from_results()
        if success_rate is not None:
            if success_rate < effective_threshold:
                print(
                    "Success rate below "
                    f"{effective_threshold}% detected in results payload: {success_rate}"
                )
                return False
            return True
        print("Warning: unable to compute success rate from results payload; falling back to CSV inspection.")

    csv_names = ("first_half.csv", "second_half.csv")
    base_dir = Path(RESULTS_DIR)
    for name in csv_names:
        path = base_dir / name
        if not path.exists():
            continue
        try:
            with path.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or 'success_rate' not in reader.fieldnames:
                    continue
                for row in reader:
                    raw_val = row.get('success_rate')
                    if raw_val is None or raw_val == '':
                        continue
                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue
                    if val < effective_threshold:
                        print(
                            f"Success rate below {effective_threshold}% detected in {name}: {val}"
                        )
                        return False
        except Exception as exc:
            print(f"Warning: unable to inspect success_rate in {path}: {exc}")
    return True

_TOKEN_SUFFIX_RE = re.compile(r"^(.*)_([0-9]+)-([0-9]+)$")

def _strip_token_suffix(filename):
    """Remove trailing _MIN-MAX suffix from filename if present."""
    name, ext = os.path.splitext(filename)
    match = _TOKEN_SUFFIX_RE.match(name)
    if match:
        return f"{match.group(1)}{ext}"
    return filename

def _get_requests_filename_base():
    base = os.environ.get('REQUESTS_FILENAME_BASE')
    if base:
        return base
    current = os.environ.get('REQUESTS_FILENAME', REQUESTS_FILENAME)
    normalized = _strip_token_suffix(current)
    os.environ['REQUESTS_FILENAME_BASE'] = normalized
    return normalized

def _mix_requests_filename(in_min, in_max):
    """Requests file of a workload profile: <base>_<in_min>-<in_max>.json.

    Mirrors the suffix that set_process_env_for_run appends to REQUESTS_FILENAME, so a profile
    reads exactly the file the harness would select for that input interval. Profiles sharing an
    input interval therefore share the file, even when their output intervals differ.
    """
    base_filename = _get_requests_filename_base()
    name, ext = os.path.splitext(base_filename)
    if not ext:
        ext = '.json'
    return f"{name}_{in_min}-{in_max}{ext}"


_WORKLOAD_MIXES_CACHE = None


def _get_workload_mixes():
    """Resolve the configured WORKLOAD_MIXES (env first, then .env).

    Parsed once per process so the tolerant parser warnings are printed a single time and every
    caller (the mix loop and the requests-file generation) sees the same mixes.
    """
    global _WORKLOAD_MIXES_CACHE
    if _WORKLOAD_MIXES_CACHE is None:
        env_val = os.environ.get('WORKLOAD_MIXES')
        if env_val is not None and str(env_val).strip():
            _WORKLOAD_MIXES_CACHE = parse_workload_mixes(env_val)
        else:
            _WORKLOAD_MIXES_CACHE = CONFIG.get('WORKLOAD_MIXES', []) or []
    return _WORKLOAD_MIXES_CACHE


def _set_additive_run_active(value):
    """Mark the process as running an additive (WORKLOAD_MIXES) execution."""
    global ADDITIVE_RUN_ACTIVE
    ADDITIVE_RUN_ACTIVE = bool(value)


def set_process_env_for_run(req_min_value, input_interval=None, output_interval=None):
    """Set environment variables in-process for a run without modifying .env.

    input_interval/output_interval can be:
    - a single int, or
    - a tuple/list (min, max)
    """
    if req_min_value is not None:
        os.environ['REQ_MIN'] = str(req_min_value)
    if input_interval is not None:
        if isinstance(input_interval, (list, tuple)) and len(input_interval) >= 2:
            os.environ['MIN_INPUT_TOKENS'] = str(input_interval[0])
            os.environ['MAX_INPUT_TOKENS'] = str(input_interval[1])
        else:
            os.environ['MIN_INPUT_TOKENS'] = str(input_interval)
            os.environ['MAX_INPUT_TOKENS'] = str(input_interval)
    if output_interval is not None:
        if isinstance(output_interval, (list, tuple)) and len(output_interval) >= 2:
            os.environ['MIN_OUTPUT_TOKENS'] = str(output_interval[0])
            os.environ['MAX_OUTPUT_TOKENS'] = str(output_interval[1])
        else:
            os.environ['MIN_OUTPUT_TOKENS'] = str(output_interval)
            os.environ['MAX_OUTPUT_TOKENS'] = str(output_interval)

    # Append min-max input interval to REQUESTS_FILENAME so downstream tools read the correct file
    if input_interval is not None:
        if os.environ.get('SERVICE_TYPE') == 'SaaS':
            os.environ['REQUESTS_FILENAME'] = REQUESTS_FILENAME
        else:
            if isinstance(input_interval, (list, tuple)) and len(input_interval) >= 2:
                in_min, in_max = int(input_interval[0]), int(input_interval[1])
            else:
                in_min = in_max = int(input_interval)
            base_filename = _get_requests_filename_base()
            name, ext = os.path.splitext(base_filename)
            if not ext:
                ext = '.json'
            os.environ['REQUESTS_FILENAME'] = f"{name}_{in_min}-{in_max}{ext}"

def run_command(command, wait=True, fail_on_error=False):
    """Run a shell command and wait for completion.

    When fail_on_error is True, raise RuntimeError on non-zero exit code.
    """
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    if wait:
        process.wait()
        if process.returncode != 0:
            message = f"Command '{command}' returned non-zero exit code: {process.returncode}"
            if fail_on_error:
                raise RuntimeError(message)
            print(f"Warning: {message}")
    return process

def run_command_capture(command):
    """Run a shell command, wait, and capture stdout/stderr."""
    print(f"Running (capture): {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Warning: Command '{command}' returned non-zero exit code: {result.returncode}")
    return result.returncode, result.stdout, result.stderr

def run_evaluation_pipeline(experiment_type):
    """Run the evaluation pipeline steps 3-7 and return throughput metric."""
    scripts_dir = Path(__file__).resolve().parent / 'requests'
    convert_to_csv_script = scripts_dir / 'convert_to_csv.py'
    analyze_metrics_script = scripts_dir / 'analyze_metrics.py'
    split_results_script = scripts_dir / 'split_results.py'
    evaluate_script = scripts_dir / 'evaluate.py'

    # Step 3: Run loadgen
    run_command(f'"{sys.executable}" -u -m fmperf.loadgen.run', fail_on_error=True)
    
    # Step 4: Change to results directory
    original_dir = os.getcwd()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.chdir(RESULTS_DIR)
    
    try:
        # Step 5: Convert to CSV
        run_command(f'"{sys.executable}" -u "{convert_to_csv_script}"')
        
        # Early metrics check before splitting results (invoke analyze_metrics.py as a script)
        early_fail = False
        median_resp_per_min = None
        try:
            code, out, err = run_command_capture(f'"{sys.executable}" -u "{analyze_metrics_script}" .')
            if code == 0:
                avg_resp = 0.0
                for line in out.splitlines():
                    if "Median responded requests per minute:" in line:
                        try:
                            avg_resp = float(line.split(":", 1)[1].strip())
                        except Exception:
                            pass
                        break
                median_resp_per_min = avg_resp
                req_min_env = os.environ.get('REQ_MIN', '0')
                try:
                    req_min_val = float(req_min_env)
                except ValueError:
                    req_min_val = 0.0
                if experiment_type != 'MIT' and avg_resp < (0.95 * req_min_val):
                    print(f"Early evaluation failure: median responded/min = {avg_resp:.3f} < 95% of REQ_MIN = {req_min_val}")
                    early_fail = True
                else:
                    print(
                        f"Early metrics check{' (MIT informational only)' if experiment_type == 'MIT' else ' passed'}: "
                        f"median responded/min = {avg_resp:.3f}, REQ_MIN = {req_min_val}"
                    )
            else:
                # Non-zero exit from analyze_metrics: log and continue normal flow
                print(f"Warning: analyze_metrics.py exited with code {code}. stderr: {err.strip()}")
        except Exception as e:
            # Don't block on metrics errors; proceed with normal flow
            print(f"Warning: analyze_metrics script invocation failed: {e}")
            early_fail = False
        
        # Step 6: Split results
        run_command(f'"{sys.executable}" -u "{split_results_script}"')
        
        # Step 7: Run evaluation (skip if early failure triggered)
        if early_fail:
            result = None
            evaluation_success = False
        elif experiment_type == 'MIT':
            # MIT experiments rely on throughput plateau detection later
            evaluation_success = True
        else:
            result = run_command(f'"{sys.executable}" -u "{evaluate_script}"', wait=True)
            # Evaluation.py returns 0 for success, non-zero for failure
            evaluation_success = (result.returncode == 0)
        
        return evaluation_success, median_resp_per_min
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def start_stage_1():
    """Initialize stage 1"""
    return 1

def end_experiment(stage, M, m, evaluation):
    """Check termination condition for stage 2 using a relative threshold based on M.

    Stops when the gap (M - m) is less than or equal to M * STOP_THRESHOLD.
    Falls back to absolute STOP_THRESHOLD if M or m are not numbers.
    """
    stop_threshold = CONFIG.get('STOP_THRESHOLD', 0.5)
    if stage == 2 and (M is not None) and (m is not None):
        try:
            relative_threshold = float(M) * float(stop_threshold)
        except Exception:
            # Fallback: treat threshold as absolute if casting fails
            relative_threshold = float(stop_threshold)

        if (M - m) <= relative_threshold:
            if evaluation:
                return True, "REQ_MIN", None  # Return REQ_MIN as result
            else:
                return True, "m", m  # Return m as result
    return False, None, None

def update_stage_1(evaluation, current_req_min, retry_count_stage1, highest_true, lowest_false):
    """Update logic for stage 1 with a retry mechanism.

    Behavior:
    - Success (TRUE): track highest TRUE; if no confirmed FALSE yet, increase REQ_MIN by multiplier; if a confirmed FALSE exists, transition to stage 2.
    - Failure (FALSE): require a double-false at the SAME REQ_MIN to confirm the FALSE bound. A single FALSE followed by TRUE must NOT start stage 2.
    - If no TRUE yet and a value is confirmed FALSE (double-false), keep decreasing to find a TRUE bound.

    Stage 2 only starts when there is at least one TRUE (lower bound) and a CONFIRMED FALSE (upper bound via double-false).

    Returns: (stage, new_req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false)
    """
    multiplier = CONFIG.get('REQ_MIN_INCREASE_MULTIPLIER', 2.0)

    if evaluation:
        # Successful evaluation: record lower bound and clear any pending failure retry
        if highest_true is None or current_req_min > highest_true:
            highest_true = current_req_min

        # Clear failure retry on success
        retry_count_stage1 = 0

        if lowest_false is None:
            # No confirmed FALSE yet → keep increasing
            # Use float multiplier, keep REQ_MIN as integer via rounding
            new_req_min = max(1, int(round(current_req_min * multiplier)))
            return 1, new_req_min, None, None, None, None, retry_count_stage1, highest_true, lowest_false
        else:
            # We have a confirmed FALSE and at least one TRUE → transition to stage 2
            M_0 = lowest_false
            m_0 = highest_true
            stage = 2
            M = M_0
            m = m_0
            new_req_min = (M + m) / 2
            return stage, new_req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false
    else:
        # Failed evaluation: require double-false to confirm upper bound
        if retry_count_stage1 == 0:
            # First failure at this REQ_MIN → retry same value to confirm
            retry_count_stage1 = 1
            return 1, current_req_min, None, None, None, None, retry_count_stage1, highest_true, lowest_false
        else:
            # Second consecutive failure at same REQ_MIN → confirmed FALSE
            retry_count_stage1 = 0
            if lowest_false is None or current_req_min < lowest_false:
                lowest_false = current_req_min

            if highest_true is None:
                # No TRUE yet: keep decreasing to find a TRUE bound
                # Use float multiplier, keep REQ_MIN as integer via rounding
                new_req_min = max(1, int(round(current_req_min / multiplier)))
                return 1, new_req_min, None, None, None, None, retry_count_stage1, highest_true, lowest_false
            else:
                # Have TRUE and confirmed FALSE → transition to stage 2
                M_0 = lowest_false
                m_0 = highest_true
                stage = 2
                M = M_0
                m = m_0
                new_req_min = (M + m) / 2
                return stage, new_req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false

def update_stage_2(evaluation, current_req_min, M, m, retry_count_stage2):
    """Update logic for stage 2 with retry mechanism (uses retry_count_stage2)."""
    if evaluation:
        m = current_req_min  # Successful evaluation: move lower bound up
        retry_count_stage2 = 0  # Reset retry count on success
    else:
        if retry_count_stage2 == 0:
            # First failure: don't update M, just increment retry count
            retry_count_stage2 = 1
        else:
            # Second consecutive failure: update M and reset retry count
            M = current_req_min
            retry_count_stage2 = 0

    new_req_min = (M + m) / 2
    return new_req_min, M, m, retry_count_stage2

def run_experiment_for_tokens(tokens, initial_req_min=None, workload_mix=None):
    """Run the complete experiment for a specific token combination.

    initial_req_min: optional initial value for REQ_MIN specific to this
    token combination. If None, defaults to 1.

    workload_mix: optional mix dict from workload_mix.parse_workload_mixes. When given this is an
    additive experiment: `tokens` is then only the mix envelope (kept for the store_results.py
    layout detection), the parent folder is named after the mix, and every request is routed to
    one of the mix profiles by the loadgen (see WORKLOAD_MIX_SPEC).
    """
    # Get environment variables at the start and store them as Python variables
    model = os.environ.get('MODEL', '')
    experiment_type = get_experiment_type()
    is_mit = (experiment_type == 'MIT')
    print(f"Experiment type: {experiment_type}")
    duration_seconds = _get_duration_seconds()
    iteration_cooldown_seconds = _get_iteration_cooldown_seconds()
    iteration_hard_limit = _get_iteration_hard_limit()
    if duration_seconds is None and is_mit:
        print("Warning: Unable to determine experiment duration; MIT throughput checks may be unavailable.")
    print(f"Iteration hard limit for this token experiment: {iteration_hard_limit}")
    
    print(f"Stored configuration - MODEL: {model}")

    # Resolve model once per token experiment to avoid repeated URL probing downstream.
    if os.environ.get('SERVICE_TYPE') == 'SaaS':
        model_from_url = 'SaaS'
    else:
        endpoint_candidates = [
            os.environ.get('URL', ''),
            os.environ.get('FMPERF_ENDPOINT_URL', ''),
            os.environ.get('ENDPOINT_URL', ''),
            _read_env_value(Path('.env'), 'URL', ''),
            _read_env_value(Path('.env'), 'FMPERF_ENDPOINT_URL', ''),
        ]
        model_from_url = ''
        for endpoint_value in endpoint_candidates:
            model_from_url = _extract_model_from_url(endpoint_value)
            if model_from_url:
                break
        if not model_from_url:
            # Last resort: endpoint value itself can still be informative.
            for endpoint_value in endpoint_candidates:
                hint = _model_hint_from_endpoint_value(endpoint_value)
                if hint:
                    model_from_url = hint
                    break
    if model_from_url:
        os.environ['MODEL_USED_RESOLVED'] = model_from_url
        print(f"Resolved model from URL metadata: {model_from_url}")
    
    # Create parent directory for this token pair
    # tokens can be [in_min,in_max,out_min,out_max] or [in,out]
    additive = workload_mix is not None
    if additive:
        # Additive run (WORKLOAD_MIXES): the parent folder is named after the mix, never after a
        # single interval, and WORKLOAD_MIX identifies the experiment in results.csv. The token
        # intervals below are the mix envelope and are only reported to store_results.py so its
        # compact-layout detection keeps working; store_results.py blanks the four
        # MIN/MAX_INPUT/OUTPUT_TOKENS columns when ADDITIVE is set. WORKLOAD_MIX_SPEC carries the
        # per-profile routing rules (intervals, alphas, requests files) to the loadgen and to
        # store_results.py, which uses it for the per-profile interval validation.
        os.environ['ADDITIVE'] = 'TRUE'
        os.environ['WORKLOAD_MIX'] = workload_mix['canonical']
        os.environ['WORKLOAD_MIX_SPEC'] = mix_spec_payload(workload_mix, _mix_requests_filename)
        parent_dir = workload_mix['parent_dir']
        in_min = workload_mix['envelope']['in_min']
        in_max = workload_mix['envelope']['in_max']
        out_min = workload_mix['envelope']['out_min']
        out_max = workload_mix['envelope']['out_max']
        input_interval = (in_min, in_max)
        output_interval = (out_min, out_max)
        interval_strs = (f"{in_min}-{in_max}", f"{out_min}-{out_max}")
        print(
            f"Additive workload mix: {workload_mix['canonical']} "
            f"(envelope {interval_strs[0]}:{interval_strs[1]}) -> parent folder {parent_dir}"
        )
    elif os.environ.get('SERVICE_TYPE') == 'SaaS':
        use_case_id = tokens[0]
        parent_dir = use_case_id
        input_interval = 'SaaS'
        output_interval = 'SaaS'
        interval_strs = (use_case_id, 'SaaS')
        in_min = in_max = 0
        out_min = out_max = 0
    elif len(tokens) >= 4:
        in_min, in_max, out_min, out_max = tokens[0], tokens[1], tokens[2], tokens[3]
        parent_dir = f"{in_min}-{in_max}_{out_min}-{out_max}"
        input_interval = (in_min, in_max)
        output_interval = (out_min, out_max)
        interval_strs = (f"{in_min}-{in_max}", f"{out_min}-{out_max}")
    else:
        in_min = in_max = tokens[0]
        out_min = out_max = tokens[1]
        parent_dir = f"{tokens[0]}_{tokens[1]}"
        input_interval = tokens[0]
        output_interval = tokens[1]
        interval_strs = (str(tokens[0]), str(tokens[1]))

    if not additive:
        # Reset the additive markers explicitly: a non-additive experiment must never inherit the
        # mix of a previous run in the same process, because the loadgen and store_results.py read
        # them from the environment.
        os.environ['ADDITIVE'] = 'FALSE'
        os.environ['WORKLOAD_MIX'] = ''
        os.environ['WORKLOAD_MIX_SPEC'] = ''
    
    # Track this execution's result folder so it can be archived at the end.
    if parent_dir and parent_dir not in CREATED_RESULT_DIRS:
        CREATED_RESULT_DIRS.append(parent_dir)
    
    # Step 1: Initialize stage 1
    stage = start_stage_1()
    req_min = initial_req_min if initial_req_min is not None else 1  # Per-combo initial value
    
    # Stage 2 variables (initialized when transitioning to stage 2)
    M_0, m_0, M, m = None, None, None, None
    
    # Bounds tracking for stage 1
    highest_true = None  # Highest req_min that yielded TRUE
    lowest_false = None  # Lowest req_min that yielded FALSE

    # Retry counters for stage 1 and stage 2
    retry_count_stage1 = 0
    retry_count_stage2 = 0
    mit_rpm_history = []
    
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0
 
    # Set process env for the initial request generation without modifying .env
    set_process_env_for_run(req_min, input_interval=input_interval, output_interval=output_interval)
    sample_file = Path(REQUESTS_DIR) / 'sample_requests.json'
    if sample_file.exists():
        sample_file.unlink()
    
    if os.environ.get('SERVICE_TYPE') == 'SaaS':
        # Create a dummy sample_requests.json if it doesn't exist to satisfy loadgen load phase
        req_path = Path(REQUESTS_DIR) / REQUESTS_FILENAME
        req_path.parent.mkdir(parents=True, exist_ok=True)
        with open(req_path, 'w', encoding='utf-8') as f:
            json.dump([{"request": {}, "expected": []}], f)
        print(f"Bypassing generation; created dummy SaaS workload: {req_path}")
    elif additive:
        # Additive run: each profile reads its own requests file. Files are cached per input
        # interval and generated with the union of the output intervals used by that input
        # interval across every configured mix (a shared file has to be able to serve all of
        # them); the length actually requested per request is chosen from the profile interval by
        # the loadgen.
        prompts_path = REQUESTS_PROMPTS_FILE.resolve()
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts dataset missing: {prompts_path}")
        for profile in workload_mix['profiles']:
            req_path = Path(REQUESTS_DIR) / _mix_requests_filename(
                profile['in_min'], profile['in_max']
            )
            if req_path.is_file():
                print(f"Found existing workload: {req_path}. Using cached file.")
                continue
            gen_out_min, gen_out_max = output_bounds_for_input(
                _get_workload_mixes(), profile['in_min'], profile['in_max']
            )
            print(
                f"Not found workload: {req_path}. Generating new workload for input tokens "
                f"{profile['in_min']}-{profile['in_max']} "
                f"(output bounds {gen_out_min}-{gen_out_max})..."
            )
            command = (
                f'"{sys.executable}" -u generate_requests.py '
                f"{profile['in_min']} {profile['in_max']} "
                f'--prompts-file "{prompts_path}" '
                f'--output "{req_path}" '
                f'--min-output {gen_out_min} --max-output {gen_out_max}'
            )
            run_command(command, wait=True)
            if req_path.is_file():
                print(f"Generated workload: {req_path}")
            else:
                raise FileNotFoundError(
                    f"Workload generation failed; expected file not found: {req_path}"
                )
    else:
        # Skip generation if interval-specific file already exists (uses REQUESTS_FILENAME with input suffix)
        req_filename = os.environ.get('REQUESTS_FILENAME', REQUESTS_FILENAME)
        req_path = Path(REQUESTS_DIR) / req_filename
        if req_path.is_file():
            print(f"Found existing workload: {req_path}. Using cached file.")
        else:
            print(f"Not found workload: {req_path}. Generating new workload for input tokens {in_min}-{in_max}...")
            prompts_path = REQUESTS_PROMPTS_FILE.resolve()
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts dataset missing: {prompts_path}")
            command = (
                f'"{sys.executable}" -u generate_requests.py {in_min} {in_max} '
                f'--prompts-file "{prompts_path}" '
                f'--output "{req_path}"'
            )
            if out_min is not None and out_max is not None:
                command += f" --min-output {out_min} --max-output {out_max}"
            run_command(command, wait=True)
            if req_path.is_file():
                print(f"Generated workload: {req_path}")
            else:
                raise FileNotFoundError(
                    f"Workload generation failed; expected file not found: {req_path}"
                )
    
    def _compute_median_response_tokens():
        """Compute (median tokens per response, total completed requests)."""
        try:
            results_path = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
            if not isinstance(rows, list):
                return None, None
            totals = {}
            worker_idxs = set()
            for r in rows:
                rid = r.get("request_idx")
                n = r.get("n_tokens", 0)
                wid = r.get("worker_idx")
                if rid is None:
                    continue
                try:
                    n_val = float(n)
                except Exception:
                    n_val = 0.0
                totals[rid] = totals.get(rid, 0.0) + n_val
                if wid is not None:
                    worker_idxs.add(wid)
            worker_count = len(worker_idxs) if len(worker_idxs) > 0 else 1
            values = [v / worker_count for v in totals.values()]
            total_completed_requests = len(totals)
            if not values:
                return 0.0, total_completed_requests
            values.sort()
            mid = len(values) // 2
            if len(values) % 2 == 1:
                return float(values[mid]), total_completed_requests
            return (values[mid - 1] + values[mid]) / 2.0, total_completed_requests
        except Exception as e:
            print(f"Warning: unable to compute median response tokens: {e}")
            return None, None

    def _confirmed_bounds():
        """Confirmed (largest TRUE, smallest FALSE) REQ_MIN bounds for the current stage.

        Stage 1 uses the bounds confirmed by update_stage_1 (highest TRUE /
        lowest FALSE); stage 2 uses the binary-search bounds m (largest TRUE) and
        M (smallest FALSE). They are read at persistence time, after the stage
        update, so an MIT iteration whose final verdict was changed by the
        success-rate or plateau checks is correctly reflected in the persisted row.
        """
        if stage == 2:
            return m, M
        return highest_true, lowest_false

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration}, Stage {stage}, INPUT_TOKENS={interval_strs[0]}, OUTPUT_TOKENS={interval_strs[1]}, REQ_MIN={req_min} ---")
        
        # Step 2: Update in-process environment for this iteration (no .env writes)
        set_process_env_for_run(req_min)
        
        # Steps 3-7: Run evaluation pipeline with stored variables, passing current stage and parent_dir
        try:
            evaluation_result, responded_per_min = run_evaluation_pipeline(
                experiment_type
            )
        except RuntimeError as exc:
            reason = str(exc)
            print(f"Controlled stop: terminating experiment early due to fatal pipeline error: {reason}")
            return {"aborted": True, "reason": reason}
        # Capture the REQ_MIN used for this evaluation before any update logic.
        # NOTE: for MIT the final verdict is only known after the success-rate and
        # plateau checks below, so the confirmed bounds are read at persistence
        # time (see _confirmed_bounds) instead of tracking raw verdicts here.
        req_min_used = req_min
        median_resp_tokens, total_completed_requests = _compute_median_response_tokens()
        requests_per_sec = None
        if duration_seconds and total_completed_requests is not None:
            try:
                requests_per_sec = total_completed_requests / float(duration_seconds)
            except Exception:
                requests_per_sec = None

        if is_mit:
            success_rate_ok = _check_success_rate_threshold(prefer_results_json=True)
            if not success_rate_ok:
                print(
                    f"MIT iteration failed due to success rate falling below {SUCCESS_RATE_THRESHOLD}%."
                )
                evaluation_result = False
            elif evaluation_result and stage == 1:
                responded_per_min = responded_per_min or (
                    requests_per_sec * 60.0 if requests_per_sec is not None else None
                )
                if responded_per_min is None:
                    print(
                        "Warning: Unable to compute responded requests per minute for MIT iteration; "
                        "skipping plateau detection for now."
                    )
                else:
                    mit_rpm_history.append(responded_per_min)
                    if len(mit_rpm_history) >= 3:
                        third_last = mit_rpm_history[-3]
                        second_last = mit_rpm_history[-2]
                        last = mit_rpm_history[-1]
                        prev_delta = second_last - third_last
                        curr_delta = last - second_last
                        threshold = max(abs(prev_delta) * MIT_PLATEAU_REL_TOL, MIT_PLATEAU_ABS_TOL)
                        if curr_delta < 0:
                            evaluation_result = False
                            print(
                                "Detected throughput regression: "
                                f"prev={second_last:.4f} rpm -> current={last:.4f} rpm (delta {curr_delta:.4f})."
                            )
                        elif abs(curr_delta) <= threshold:
                            evaluation_result = False
                            print(
                                "Detected MIT plateau using last three iterations: "
                                f"prev Δ={prev_delta:.4f}, current Δ={curr_delta:.4f}, "
                                f"threshold={threshold:.4f}."
                            )

        print(f"Evaluation result: {'Success' if evaluation_result else 'Failure'}")
        if stage == 2 and not evaluation_result:
            print(f"Retry count: {retry_count_stage2}")

        # When this is set, persist artifacts for the current iteration first,
        # then return the recorded value after store_results.py runs.
        stop_after_persist = False
        persist_return_value = None
        termination_reason = ''
        req_min_for_store = req_min_used
        evaluation_for_store = evaluation_result
        binary_distance_abs = ''
        binary_distance_rel = ''

        # Step 8: Check termination condition (for stage 2)
        if stage == 2:
            should_end, result_type, result_value = end_experiment(stage, M, m, evaluation_result)
            if should_end:
                if result_type == "REQ_MIN":
                    print(f"\nExperiment completed successfully! Optimal REQ_MIN = {req_min}")
                    stop_after_persist = True
                    persist_return_value = req_min
                else:
                    print(f"\nExperiment completed! Result m = {result_value}")
                    stop_after_persist = True
                    persist_return_value = result_value

        if not stop_after_persist and iteration >= iteration_hard_limit:
            if stage == 1:
                stop_after_persist = True
                evaluation_for_store = False
                termination_reason = (
                    f"FAILED_STAGE1_ITERATION_LIMIT_EXCEEDED(limit={iteration_hard_limit}, iteration={iteration})"
                )
                persist_return_value = {
                    "failed": True,
                    "reason": "stage1_iteration_hard_limit_exceeded",
                    "iteration": iteration,
                    "limit": iteration_hard_limit,
                }
                print(
                    "Iteration hard limit reached in Stage 1. "
                    "Ending current token experiment as failed."
                )
            elif stage == 2:
                stop_after_persist = True
                termination_reason = (
                    f"STOPPED_STAGE2_ITERATION_LIMIT_EXCEEDED(limit={iteration_hard_limit}, iteration={iteration})"
                )
                largest_true, smallest_false = _confirmed_bounds()
                if largest_true is not None:
                    req_min_for_store = largest_true
                persist_return_value = largest_true

                if (largest_true is not None) and (smallest_false is not None):
                    try:
                        abs_gap = float(smallest_false) - float(largest_true)
                        binary_distance_abs = f"{abs_gap:.6f}".rstrip('0').rstrip('.')
                    except Exception:
                        binary_distance_abs = ''

                if (M_0 is not None) and (m_0 is not None) and binary_distance_abs != '':
                    try:
                        initial_gap = float(M_0) - float(m_0)
                        if initial_gap != 0:
                            rel_gap = float(binary_distance_abs) / initial_gap
                            binary_distance_rel = f"{rel_gap:.6f}".rstrip('0').rstrip('.')
                    except Exception:
                        binary_distance_rel = ''

                print(
                    "Iteration hard limit reached in Stage 2. "
                    f"Returning largest TRUE REQ_MIN: {largest_true}"
                )
        
        # Steps 9-10: Update stage variables
        if not stop_after_persist:
            if stage == 1:
                (stage, req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false) = update_stage_1(
                    evaluation_result, req_min, retry_count_stage1, highest_true, lowest_false
                )
                # If we transitioned to stage 2, reset stage-2 retry counter and log bounds
                if stage == 2:
                    retry_count_stage2 = 0
                    # Reset MIT history to avoid stage-1 trend checks leaking into stage-2 binary search.
                    if is_mit:
                        mit_rpm_history = []
                    print(f"Transitioned to Stage 2: highest TRUE = {highest_true}, lowest FALSE = {lowest_false}")
            elif stage == 2:
                req_min, M, m, retry_count_stage2 = update_stage_2(
                    evaluation_result, req_min, M, m, retry_count_stage2
                )

        # End-of-iteration logging: response and throughput metrics
        print("--- Iteration summary ---")
        if median_resp_tokens is not None:
            print(f"Median tokens per response: {median_resp_tokens:.3f}")
        if total_completed_requests is not None:
            print(f"Completed requests: {total_completed_requests}")
        if responded_per_min is not None:
            print(f"Responded requests per minute (median): {responded_per_min:.3f}")
        if requests_per_sec is not None:
            print(f"Requests per second: {requests_per_sec:.4f}")

        # Persist results with explicit values, including the printed median
        try:
            original_dir2 = os.getcwd()
            os.makedirs(RESULTS_DIR, exist_ok=True)
            os.chdir(RESULTS_DIR)
            store_results_script = Path(__file__).resolve().parent / 'requests' / 'store_results.py'
            # store_results.py reads EXPERIMENT_TYPE from its own environment, so export the
            # resolved value (works when the type comes from .env or from the default). It is
            # deliberately NOT passed as an argv entry: store_results.py parses the positional
            # order (model, stage, parent_dir, in_range, out_range, req_min, evaluation, ...)
            # and detects that format from the third argument, so extra leading arguments
            # silently switch it to its legacy parser and corrupt results.csv.
            os.environ['EXPERIMENT_TYPE'] = get_experiment_type()
            evaluation_flag = "TRUE" if evaluation_for_store else "FALSE"
            median_str = f"{median_resp_tokens:.3f}" if isinstance(median_resp_tokens, (int, float)) else (str(median_resp_tokens) if median_resp_tokens is not None else '')
            # Confirmed bounds are read after the stage update so that the row
            # reflects the final verdict of this iteration (MIT included).
            confirmed_largest_true, confirmed_smallest_false = _confirmed_bounds()
            largest_true_str = _format_bound_number(confirmed_largest_true)
            smallest_false_str = _format_bound_number(confirmed_smallest_false)
            finished_flag_str = 'TRUE' if stop_after_persist else 'FALSE'
            # store_results.py derives prompt aggregates from the requests payload.
            store_args = [
                sys.executable, "-u", str(store_results_script),
                str(model), str(stage), str(parent_dir),
                str(interval_strs[0]), str(interval_strs[1]), str(req_min_for_store), str(evaluation_flag), str(median_str),
                str(os.environ.get('MODEL_USED_RESOLVED', '')),
                str(termination_reason),
                str(binary_distance_abs),
                str(binary_distance_rel),
                largest_true_str,
                smallest_false_str,
                finished_flag_str,
            ]
            # Contract self-check for the argv order above: store_results.py infers the layout
            # from these positions, so a stray argument before model/stage/parent_dir silently
            # shifts every results.csv column. Warn, never abort the experiment.
            store_positional = store_args[3:]  # strip sys.executable, -u and the script path
            if not _store_args_contract_ok(store_positional):
                print(
                    "Warning: store_results.py argv contract mismatch — expected "
                    "(model, stage, parent_dir, in_range, out_range, req_min, evaluation, ...); "
                    "results.csv columns will be misaligned. Append new arguments at the end, "
                    "never in the middle."
                )
            print("Running (args):", " ".join(store_args))
            subprocess.run(store_args)
        finally:
            os.chdir(original_dir2)
        
        if stop_after_persist:
            return persist_return_value

        # Wait before the next iteration to allow system cooldown.
        if iteration < max_iterations and iteration_cooldown_seconds > 0:
            print(
                f"Cooling down for {iteration_cooldown_seconds:g}s before next iteration..."
            )
            time.sleep(iteration_cooldown_seconds)
    
    print(f"\nReached maximum iterations ({max_iterations}). Stopping.")
    return None

def _archive_execution_results():
    """Move this execution's per-token result folders into a single archive folder.

    The archive is created inside the results directory and named
    Experiment_[EXPERIMENT_TYPE]_[YYYY-MM-DD_HH-MM-SS], where the timestamp reflects when
    the automation finished (i.e., when this function runs). Additive executions
    (WORKLOAD_MIXES) use Experiment_MIX_[EXPERIMENT_TYPE]_[YYYY-MM-DD_HH-MM-SS] instead.
    """
    if not CREATED_RESULT_DIRS:
        return

    root_dir = Path(__file__).resolve().parent
    results_dir = Path(
        os.environ.get('RESULTS_DIR') or _read_env_value(root_dir / '.env', 'RESULTS_DIR', 'results')
    )
    if not results_dir.is_absolute():
        results_dir = root_dir / results_dir

    experiment_type = re.sub(r'[^0-9A-Za-z_-]', '_', get_experiment_type())
    finished_at = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    archive_prefix = 'Experiment_MIX' if ADDITIVE_RUN_ACTIVE else 'Experiment'
    archive_dir = results_dir / f"{archive_prefix}_{experiment_type}_{finished_at}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for name in CREATED_RESULT_DIRS:
        src = results_dir / name
        if src.is_dir():
            shutil.move(str(src), str(archive_dir / name))
            moved += 1
        else:
            print(f"Warning: expected result folder not found: {src}")
    print(f"Archived {moved} result folder(s) into {archive_dir}")


def _run_additive_experiments(workload_mixes):
    """Run one additive experiment per WORKLOAD_MIXES entry, sequentially.

    Each mix is a full stage-1/stage-2 run whose requests are routed to the mix profiles by the
    loadgen. `REQ_MIN_START[idx]` seeds the idx-th mix (the last value is reused when the list is
    shorter), matching the TOKENS_LIST behaviour. TOKENS_LIST is not used in this mode.
    """
    _set_additive_run_active(True)
    req_min_starts = CONFIG.get('REQ_MIN_START', [1])
    results = {}
    print(f"Found {len(workload_mixes)} additive workload mix(es) in WORKLOAD_MIXES.")
    print("TOKENS_LIST is ignored for this execution.")

    for idx, mix in enumerate(workload_mixes):
        print(f"\n{'='*60}")
        print(f"Starting additive experiment for WORKLOAD_MIX={mix['canonical']}")
        print(f"{'='*60}")

        # Pick initial REQ_MIN by index; if not enough values, use the last one
        if req_min_starts:
            initial_req_min = req_min_starts[idx] if idx < len(req_min_starts) else req_min_starts[-1]
        else:
            initial_req_min = 1

        # The mix envelope is passed as the token interval so store_results.py keeps detecting its
        # compact argv layout; the persisted MIN/MAX_*_TOKENS columns are blank for additive rows.
        envelope = mix['envelope']
        tokens = [envelope['in_min'], envelope['in_max'], envelope['out_min'], envelope['out_max']]
        os.environ['SERVICE_TYPE'] = 'LLM'
        result = run_experiment_for_tokens(tokens, initial_req_min, workload_mix=mix)
        results[mix['canonical']] = result

        if isinstance(result, dict) and result.get("aborted"):
            print("Aborted remaining experiments after fatal pipeline error.")
            break

        print(f"\nCompleted additive experiment for WORKLOAD_MIX={mix['canonical']}")
        print(f"Result: {result}")

    print(f"\n{'='*60}")
    print("ALL ADDITIVE EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    for mix_key, result in results.items():
        print(f"WORKLOAD_MIX {mix_key}: {result}")

    # Move this execution's results into their final archive folder (Experiment_MIX_*).
    _archive_execution_results()
    return results


def main():
    service_type = CONFIG.get('SERVICE_TYPE', 'LLM')
    workload_mixes = _get_workload_mixes()

    if workload_mixes and service_type == 'SaaS':
        print(
            "Warning: WORKLOAD_MIXES is ignored in SaaS mode; the use cases of USE_CASES_YAML "
            "are run instead."
        )
        workload_mixes = []
    if workload_mixes:
        return _run_additive_experiments(workload_mixes)
    
    if service_type == 'SaaS':
        yaml_path = CONFIG.get('USE_CASES_YAML')
        use_cases = load_use_cases_from_yaml(yaml_path)
        if not use_cases:
            print("Error: No use cases found for SaaS mode.")
            return {}
        
        req_min_starts = CONFIG.get('REQ_MIN_START', [1])
        results = {}
        
        for idx, uc in enumerate(use_cases):
            use_case_id = uc.get('id', f'usecase_{idx}')
            print(f"\n{'='*60}")
            print(f"Starting experiment for SaaS Use Case: {use_case_id}")
            print(f"{'='*60}")
            
            if req_min_starts:
                initial_req_min = req_min_starts[idx] if idx < len(req_min_starts) else req_min_starts[-1]
            else:
                initial_req_min = 1
            
            os.environ['ACTIVE_USE_CASE_ID'] = use_case_id
            os.environ['SERVICE_TYPE'] = 'SaaS'
            os.environ['USE_CASES_YAML'] = str(yaml_path)
            
            # Map parameters to run_experiment_for_tokens
            result = run_experiment_for_tokens((use_case_id, 'SaaS'), initial_req_min)
            results[use_case_id] = result
            
            if isinstance(result, dict) and result.get("aborted"):
                print("Aborted remaining experiments after fatal pipeline error.")
                break
                
            print(f"\nCompleted experiment for SaaS Use Case: {use_case_id}")
            print(f"Result: {result}")
        
        print(f"\n{'='*60}")
        print("ALL SaaS EXPERIMENTS COMPLETED")
        print(f"{'='*60}")
        for uc_id, result in results.items():
            print(f"Use Case {uc_id}: {result}")
        _archive_execution_results()
        return results
    else:
        input_output_tokens = CONFIG.get('TOKENS_LIST', [])
        req_min_starts = CONFIG.get('REQ_MIN_START', [1])
        results = {}
        
        for idx, tokens in enumerate(input_output_tokens):
            print(f"\n{'='*60}")
            print(f"Starting experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
            print(f"{'='*60}")
            
            # Pick initial REQ_MIN by index; if not enough values, use the last one
            if req_min_starts:
                initial_req_min = req_min_starts[idx] if idx < len(req_min_starts) else req_min_starts[-1]
            else:
                initial_req_min = 1
            
            os.environ['SERVICE_TYPE'] = 'LLM'
            result = run_experiment_for_tokens(tokens, initial_req_min)
            results[f"{tokens[0]}_{tokens[1]}"] = result

            if isinstance(result, dict) and result.get("aborted"):
                print("Aborted remaining experiments after fatal pipeline error.")
                break
            
            print(f"\nCompleted experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
            print(f"Result: {result}")
        
        print(f"\n{'='*60}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*60}")
        for token_combo, result in results.items():
            print(f"Tokens {token_combo}: {result}")
        
        # Move this execution's results into their final archive folder.
        _archive_execution_results()
        
        return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"Final results: {results}")
    except Exception as exc:
        print(f"Controlled stop: {exc}")

