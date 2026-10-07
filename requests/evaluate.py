import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.weightstats import ttost_ind
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def _load_env(path: Path) -> dict:
    env = {}
    try:
        with path.open('r', encoding='utf-8') as f:
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


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV = _load_env(ROOT_DIR / '.env')
RESULTS_DIR = ENV.get('RESULTS_DIR') or os.environ.get('RESULTS_DIR', 'results')
RESULTS_PATH = Path(RESULTS_DIR)
if not RESULTS_PATH.is_absolute():
    RESULTS_PATH = ROOT_DIR / RESULTS_PATH

SUCCESS_RATE_THRESHOLD = float(os.environ.get('SUCCESS_RATE_THRESHOLD', '95.0'))

# Per-request columns written by requests/convert_to_csv.py (see its CSV_COLUMNS).
# These replaced the legacy `complete_response_time` / `received_timestamp` names;
# split_results.py still accepts legacy CSVs, and the contract test in
# fmperf/tests/test_output_csv_contract.py guards this agreement.
METRICS = ["request_duration_ms"]
TIMESTAMP_COLUMN = "full_response_received_at_utc"

# Function definitions
def welch_ttest(x: np.ndarray, y: np.ndarray):
    """Test t de Welch con gestión de NaN."""
    stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return stat, p

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cálculo del tamaño del efecto de Cohen (d)."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)

def bootstrap_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
):
    """Bootstrap simple (remuestreo con reemplazo). Devuelve
    la media y el IC 95 % de la diferencia de medias."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.mean(xb) - np.mean(yb))
    diffs = np.asarray(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), (ci_low, ci_high), diffs

def equivalence_test(
    x: np.ndarray,
    y: np.ndarray,
    margin: float,
) -> float:
    """TOST con margen de equivalencia simétrico (mismo dominio de la métrica)."""
    p = ttost_ind(x, y, -margin, margin, usevar="unequal")[0]
    return p   # p < α ⇒ equivalencia aceptada

if __name__ == "__main__":
    n_boot = 10000
    equiv = True
    EQUIV_MARGIN = 0.02
    metrics = METRICS
    timestamp_column = TIMESTAMP_COLUMN
    
    # Define the column name for the statistical difference flag
    STAT_DIFF_COL = "no_statistical_difference_overall"

    # --- Data Loading Functions ---
    def load_dataset(path: Path) -> pd.DataFrame:
        """Carga un CSV y convierte la columna de timestamp a datetime."""
        df = pd.read_csv(path)
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce', utc=True)
        return df

    def collect_datasets() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Timestamp], Dict[str, pd.Timestamp]]:
        """
        Busca los dos archivos CSV en el directorio actual y devuelve:
            datasets    : dict nombre → DataFrame
            min_ts      : dict nombre → min timestamp
            max_ts      : dict nombre → max timestamp
        """
        datasets: Dict[str, pd.DataFrame] = {}
        min_ts_dict: Dict[str, pd.Timestamp] = {}
        max_ts_dict: Dict[str, pd.Timestamp] = {}

        filenames = [
            "first_half.csv",
            "second_half.csv"
        ]
        
        for fname in filenames:
            csv_path = RESULTS_PATH / fname
            if not csv_path.exists():
                print(f"File {fname} not found in current directory")
                continue
            try:
                df = load_dataset(csv_path)
            except ValueError as e:
                print(f"Error loading {fname}: {e}")
                continue
            datasets[fname] = df
            min_ts_dict[fname] = df[timestamp_column].iloc[0]
            max_ts_dict[fname] = df[timestamp_column].iloc[-1]

        return datasets, min_ts_dict, max_ts_dict

    # --- Interpretation Function ---
    from pandas.api.types import is_datetime64_any_dtype as _is_dt64
    from zoneinfo import ZoneInfo

    DEFAULT_TZ = ZoneInfo("Europe/Madrid")

    def fmt(ts) -> str:
        """Devuelve el timestamp en formato legible: 'YYYY-MM-DD HH:MM:SS CET/CEST'."""
        if not _is_dt64(ts):
            ts = pd.to_datetime(ts, utc=True, errors="coerce")

        if ts is pd.NaT:
            return "N/D"

        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert(DEFAULT_TZ)

        return ts.strftime("%Y-%m-%d %H:%M:%S %Z")

    def interpret_result(
        p_welch: float,
        cohen_d: float,
        boot_diff: float,
        boot_ci_lo: float,
        boot_ci_hi: float,
        p_tost: float | None = None,
        alpha: float = 0.1,
    ) -> str:
        """Returns a verbal interpretation of a Welch two-sample comparison."""
        if p_welch < alpha:
            sig_txt = "statistically significant (p < {:.3f})".format(alpha)
        else:
            sig_txt = "not statistically significant (p ≥ {:.3g})".format(alpha)

        if abs(cohen_d) < 0.2:
            mag = "negligible"
        elif abs(cohen_d) < 0.5:
            mag = "small"
        elif abs(cohen_d) < 0.8:
            mag = "medium"
        else:
            mag = "large"
        direction = "higher" if boot_diff > 0 else "lower"

        if boot_ci_lo <= 0 <= boot_ci_hi:
            ci_txt = "The 95 % bootstrap CI includes zero so the sign of the difference is uncertain."
        else:
            ci_txt = f"The 95 % bootstrap CI is strictly confirming the {direction} mean in group 1."

        if p_tost is not None:
            if p_tost < alpha:
                tost_txt = f"Additionally, the TOST indicates practical equivalence (p < {alpha})."
            else:
                tost_txt = f"TOST does not support equivalence (p ≥ {alpha})."
        else:
            tost_txt = ""

        summary = f"The difference is {sig_txt}; the observed effect is {mag}."
        if tost_txt:
            summary += "  " + tost_txt

        return summary

    # --- Comparison Function ---
    from itertools import combinations

    def compare_pair(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        metrics: List[str],
        margin: float | None = None,
        match_sizes: bool = False,
        random_state: int = 42,
        fname: str = "Dataset 1",
        fname2: str = "Dataset 2"
    ) -> List[Dict]:
        """Compara las métricas de `df1` y `df2`."""
        rs = np.random.default_rng(random_state)
        records: List[Dict] = []

        for metric in metrics:
            if metric not in df1.columns or metric not in df2.columns:
                continue

            x = df1[metric].dropna().values
            y = df2[metric].dropna().values
            if x.size == 0 or y.size == 0:
                continue

            if match_sizes and x.size != y.size:
                if x.size < y.size:
                    y = rs.choice(y, size=x.size, replace=False)
                else:
                    x = rs.choice(x, size=y.size, replace=False)

            stat, p_val = welch_ttest(x, y)
            d_val = cohens_d(x, y)
            diff_bs, ci, _ = bootstrap_mean_diff(x, y)
            p_tost = equivalence_test(x, y, margin) if margin is not None else None
            
            alpha = 0.1
            no_statistical_difference_for_pair = p_val >= alpha

            records.append({
                "file1": fname,
                "file2": fname2,
                "metric": metric,
                "n_win1": len(x),
                "n_win2": len(y),
                "t_stat": stat,
                "p_welch": p_val,
                "cohen_d": d_val,
                "boot_diff": diff_bs,
                "boot_ci_lo": ci[0],
                "boot_ci_hi": ci[1],
                "p_tost": p_tost,
                "no_statistical_difference": no_statistical_difference_for_pair,
                "first_higher_than_second": diff_bs > 0,  # True if first has higher response time
                "interprentation": interpret_result(
                    p_welch=p_val,
                    cohen_d=d_val,
                    boot_diff=diff_bs,
                    boot_ci_lo=ci[0],
                    boot_ci_hi=ci[1],
                    p_tost=p_tost,
                ),
            })

        return records

    # ───────────────────────────── Main execution logic ───────────────────────────────────
    print(f"Loading datasets from {RESULTS_PATH}...")
    
    # Collect datasets from current directory
    datasets, min_ts, max_ts = collect_datasets()

    if len(datasets) < 2:
        print("Error: Need at least two datasets (first_half.csv and second_half.csv) in RESULTS_DIR")
        sys.exit(1)  # Exit with error code

    # Check success_rate in each dataset
    skip_statistical_analysis_due_to_success_rate = False
    experiment_has_no_statistical_difference_overall = True
    
    for name, df in datasets.items():
        if "success_rate" in df.columns:
            if (df["success_rate"] < SUCCESS_RATE_THRESHOLD).any():
                print(f"Success rate below {SUCCESS_RATE_THRESHOLD}% in {name}")
                experiment_has_no_statistical_difference_overall = False
                skip_statistical_analysis_due_to_success_rate = True
                break

    if skip_statistical_analysis_due_to_success_rate:
        # If any dataset has success_rate < SUCCESS_RATE_THRESHOLD %, force overall result to FALSE
        final_result = False
        print(f"Overall {STAT_DIFF_COL}: {final_result} (due to low success rate)")
    else:
        # Perform statistical comparisons
        global_min_ts = min(min_ts.values())
        global_max_ts = max(max_ts.values())

        # Get all combinations of datasets
        dataset_names = list(datasets.keys())
        pairwise_results = []
        
        for f1, f2 in combinations(dataset_names, 2):
            comparison_results = compare_pair(
                datasets[f1],
                datasets[f2],
                metrics,
                margin=EQUIV_MARGIN,
                match_sizes=False,
                fname=f1,
                fname2=f2
            )
            pairwise_results.extend(comparison_results)

            # Check if any comparison shows statistical significance
            for record in comparison_results:
                if not record["no_statistical_difference"]:
                    experiment_has_no_statistical_difference_overall = False
                    break

        # Display results
        if pairwise_results:
            results_df = pd.DataFrame(pairwise_results)
            print("\nDetailed comparison results:")
            print(results_df.to_string())
            
            # NEW LOGIC: Return TRUE if no statistical differences OR first has higher response time
            first_has_higher_response_time = any(
                record["first_higher_than_second"] 
                for record in pairwise_results 
                if record["file1"] == "first_half.csv" and record["file2"] == "second_half.csv"
            )
            
            # Modified condition: TRUE if no statistical differences OR first has higher response time
            final_result = experiment_has_no_statistical_difference_overall or first_has_higher_response_time
            
            print(f"\nOverall {STAT_DIFF_COL}: {final_result}")
            print(f"  - No statistical differences: {experiment_has_no_statistical_difference_overall}")
            print(f"  - First 10 minutes have higher response time: {first_has_higher_response_time}")
        else:
            print("No valid comparisons could be performed")
            final_result = experiment_has_no_statistical_difference_overall
            print(f"Overall {STAT_DIFF_COL}: {final_result} (no comparable pairs)")

    # Add the statistical difference flag to any base CSV files
    for csv_file in RESULTS_PATH.glob("*.csv"):
        if csv_file.name not in ["first_half.csv", "second_half.csv"]:
            try:
                base_df = pd.read_csv(csv_file)
                base_df["no_statistical_difference"] = final_result
                base_df.to_csv(csv_file, index=False)
                print(f"Added 'no_statistical_difference = {final_result}' to {csv_file.name}")
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")

    # Return appropriate exit code based on the result
    if final_result:
        print("\nEvaluation result: TRUE (no statistical differences OR first 10 minutes have higher response time)")
        sys.exit(0)  # Success exit code
    else:
        print("\nEvaluation result: FALSE (statistical differences exist AND first 10 minutes don't have higher response time)")
        sys.exit(1)  # Error exit code
