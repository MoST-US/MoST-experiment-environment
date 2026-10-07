import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Load minutes from .env (root of workspace)
def _load_env(path):
    env = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    except Exception:
        pass
    return env

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(ROOT_DIR, '.env')
ENV = _load_env(ENV_PATH)
try:
    BUFFER_SECONDS = int(ENV.get('FILTER_BUFFER', '300'))
except ValueError:
    BUFFER_SECONDS = 300

RESULTS_DIR = ENV.get('RESULTS_DIR') or os.environ.get('RESULTS_DIR', 'results')
RESULTS_PATH = Path(RESULTS_DIR)
if not RESULTS_PATH.is_absolute():
    RESULTS_PATH = Path(ROOT_DIR) / RESULTS_PATH
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Per-request columns written by requests/convert_to_csv.py (see CSV_COLUMNS there).
# These replaced the legacy `received_timestamp` / `complete_response_time` names.
TIMESTAMP_COLUMN = 'full_response_received_at_utc'
DURATION_COLUMN = 'request_duration_ms'


def _load_timestamp_column(df):
    """Return ``df`` plus the name of its parsed, tz-aware timestamp column.

    convert_to_csv.py writes ``full_response_received_at_utc`` as an ISO-8601 string
    (e.g. "2026-10-07T12:34:56.789000+00:00"). Legacy files used
    ``received_timestamp`` in the "%Y%m%dT%H%M%S" form, so both spellings are accepted.
    """
    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(
            df[TIMESTAMP_COLUMN], errors='coerce', utc=True
        )
        return df, TIMESTAMP_COLUMN
    if 'received_timestamp' in df.columns:
        # Legacy CSVs stored the compact "%Y%m%dT%H%M%S" form.
        df['received_timestamp'] = pd.to_datetime(
            df['received_timestamp'], format='%Y%m%dT%H%M%S', errors='coerce', utc=True
        )
        return df, 'received_timestamp'
    raise KeyError(
        f"Missing timestamp column '{TIMESTAMP_COLUMN}'. The CSV must come from "
        "requests/convert_to_csv.py"
    )


def process_experiment_data(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime
    df, ts_col = _load_timestamp_column(df)
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    # Calculate total experiment duration
    start_time = df[ts_col].min()
    end_time = df[ts_col].max()
    total_duration = end_time - start_time
    
    print(f"Experiment started at: {start_time}")
    print(f"Experiment ended at: {end_time}")
    print(f"Total duration: {total_duration}")
    
    # Remove first and last N seconds (from .env: FILTER_BUFFER)
    filtered_start = start_time + timedelta(seconds=BUFFER_SECONDS)
    filtered_end = end_time - timedelta(seconds=BUFFER_SECONDS)
    
    filtered_df = df[
        (df[ts_col] >= filtered_start) & 
        (df[ts_col] <= filtered_end)
    ].copy().reset_index(drop=True)
    
    print(f"\nAfter removing first and last {BUFFER_SECONDS} minutes:")
    print(f"Filtered start: {filtered_start}")
    print(f"Filtered end: {filtered_end}")
    print(f"Filtered duration: {filtered_end - filtered_start}")
    print(f"Records in filtered data: {len(filtered_df)}")
    
    # Split filtered window into two equal halves
    filtered_duration = filtered_end - filtered_start
    split_time = filtered_start + (filtered_duration / 2)
    
    first_half = filtered_df[
        filtered_df[ts_col] < split_time
    ].copy().reset_index(drop=True)
    
    second_half = filtered_df[
        filtered_df[ts_col] >= split_time
    ].copy().reset_index(drop=True)
    
    print(f"\nSplit midpoint: {split_time}")
    print(f"First half: {len(first_half)} records")
    print(f"Second half: {len(second_half)} records")
    
    # Save to new CSV files
    first_half_path = RESULTS_PATH / 'first_half.csv'
    second_half_path = RESULTS_PATH / 'second_half.csv'
    first_half.to_csv(first_half_path, index=False)
    second_half.to_csv(second_half_path, index=False)
    
    print(f"\nFiles created:")
    print(f"- {first_half_path} ({len(first_half)} records)")
    print(f"- {second_half_path} ({len(second_half)} records)")
    
    return first_half, second_half

def _resolve_duration_column(df):
    if DURATION_COLUMN in df.columns:
        return DURATION_COLUMN
    if 'complete_response_time' in df.columns:
        return 'complete_response_time'
    return None


def _print_period_stats(label, period):
    column = _resolve_duration_column(period)
    if column is None:
        print(f"\n{label}: response-time column not found; skipping statistics")
        return
    print(f"\n{label}:")
    print(f"Response time - Min: {period[column].min():.2f}, "
          f"Max: {period[column].max():.2f}, "
          f"Mean: {period[column].mean():.2f}")


def main():
    # Process the data
    first_period, second_period = process_experiment_data(RESULTS_PATH / 'output.csv')

    # Display some statistics
    _print_period_stats("First period statistics", first_period)
    _print_period_stats("Second period statistics", second_period)


if __name__ == '__main__':
    main()
