#!/usr/bin/env python3
"""Convert token-level load-test results into one CSV row per request."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CSV_COLUMNS = (
    "request_idx",
    "request_sent_at_utc",
    "first_token_at_utc",
    "full_response_received_at_utc",
    "successful_request",
    "records_in_request",
    "response_token_count",
    "input_token_count",
    "output_token_count",
    "first_token_ms",
    "request_duration_ms",
    "token_generation_duration_sum_ms",
    "token_generation_duration_avg_ms",
    "token_generation_duration_min_ms",
    "token_generation_duration_max_ms",
    "token_generation_duration_stdev_ms",
    "token_generation_duration_p95_ms",
    "inter_token_interval_avg_ms",
    "inter_token_interval_stdev_ms",
    "tokens_per_second",
    "worker_idx",
    "exp_num_users",
    "workload_profile",
    "exclude",
    "consistent",
    "error",
    "response_char_count",
    "success_rate",
)


def timestamp_to_iso(value: Any) -> str:
    if value is None:
        return ""
    timestamp = float(value)
    magnitude = abs(timestamp)
    if magnitude >= 1e17:
        timestamp /= 1e9
    elif magnitude >= 1e14:
        timestamp /= 1e6
    elif magnitude >= 1e11:
        timestamp /= 1e3
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    )


def number(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    return (
        statistics.quantiles(values, n=100, method="inclusive")[int(fraction * 100) - 1]
        if len(values) > 1
        else values[0]
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)
    records = document.get("results") if isinstance(document, dict) else document
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Expected a JSON list or an object with a list named 'results'.")
    return records


def load_input_tokens(path: Path) -> dict[tuple[Any, Any], int]:
    """Load runtime request token counts; missing files preserve legacy behavior."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as stream:
        document = json.load(stream)
    entries = document.get("requests") if isinstance(document, dict) else document
    if not isinstance(entries, list):
        raise ValueError(f"Expected {path} to contain a list named 'requests'.")

    counts: dict[tuple[Any, Any], int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        count = entry.get("input_token_count")
        if not isinstance(count, (int, float)) or isinstance(count, bool):
            continue
        counts[(entry.get("worker_idx"), entry.get("request_idx"))] = int(count)
    return counts


def aggregate_requests(
    records: Iterable[dict[str, Any]],
    input_tokens: dict[tuple[Any, Any], int] | None = None,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for position, record in enumerate(records):
        request_id = record.get("request_idx", f"missing-{position}")
        grouped.setdefault((record.get("worker_idx"), request_id), []).append(record)

    input_tokens = input_tokens or {}
    rows: list[dict[str, Any]] = []
    for (worker_id, request_id), request_records in grouped.items():
        timed_records = [
            (float(record["timestamp"]), float(record["duration_ms"]))
            for record in request_records
            if isinstance(record.get("timestamp"), (int, float))
            and isinstance(record.get("duration_ms"), (int, float))
        ]
        timed_records.sort(key=lambda item: item[0])
        timestamps = [item[0] for item in timed_records]
        durations = [item[1] for item in timed_records]
        first_timestamp = timestamps[0] if timestamps else None
        last_timestamp = timestamps[-1] if timestamps else None
        first_token_ms = durations[0] if durations else float("nan")
        wall_duration_ms = (
            (last_timestamp - first_timestamp) / 1e6 + first_token_ms
            if first_timestamp is not None
            and last_timestamp is not None
            and not math.isnan(first_token_ms)
            else float("nan")
        )
        request_start = (
            first_timestamp - first_token_ms * 1e6
            if first_timestamp is not None and not math.isnan(first_token_ms)
            else None
        )
        token_counts = number(record.get("n_tokens") for record in request_records)
        texts = [
            record.get("response", {}).get("text", "")
            for record in request_records
            if isinstance(record.get("response"), dict)
        ]
        ok_values = [record.get("ok") for record in request_records]
        generation_intervals = [
            timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))
        ]
        durations_ms = durations
        row: dict[str, Any] = {
            "request_idx": request_id,
            "request_sent_at_utc": timestamp_to_iso(request_start),
            "first_token_at_utc": timestamp_to_iso(first_timestamp),
            "full_response_received_at_utc": timestamp_to_iso(last_timestamp),
            "successful_request": all(value is True for value in ok_values),
            "records_in_request": len(request_records),
            "response_token_count": sum(token_counts),
            "input_token_count": input_tokens.get((worker_id, request_id), ""),
            "output_token_count": sum(token_counts),
            "first_token_ms": first_token_ms,
            "request_duration_ms": wall_duration_ms,
            "token_generation_duration_sum_ms": sum(durations_ms),
            "token_generation_duration_avg_ms": statistics.mean(durations_ms) if durations_ms else float("nan"),
            "token_generation_duration_min_ms": min(durations_ms, default=float("nan")),
            "token_generation_duration_max_ms": max(durations_ms, default=float("nan")),
            "token_generation_duration_stdev_ms": statistics.stdev(durations_ms) if len(durations_ms) > 1 else float("nan"),
            "token_generation_duration_p95_ms": percentile(durations_ms, 0.95),
            "inter_token_interval_avg_ms": statistics.mean(generation_intervals) / 1e6 if generation_intervals else float("nan"),
            "inter_token_interval_stdev_ms": statistics.stdev(generation_intervals) / 1e6 if len(generation_intervals) > 1 else float("nan"),
            "tokens_per_second": sum(token_counts) / (wall_duration_ms / 1000) if wall_duration_ms > 0 else float("nan"),
            "worker_idx": worker_id if worker_id is not None else "",
            "exp_num_users": request_records[0].get("exp_num_users", ""),
            "workload_profile": request_records[0].get("workload_profile", ""),
            "exclude": request_records[0].get("exclude", ""),
            "consistent": all(record.get("consistent") is True for record in request_records),
            "error": next((record.get("error") for record in request_records if record.get("error") not in (None, "", "None")), ""),
            "response_char_count": len("".join(texts)),
        }
        if include_text:
            row["response_text"] = "".join(texts)
        rows.append(row)
    success_rate = (
        sum(1 for row in rows if row["successful_request"]) / len(rows) * 100
        if rows
        else 0
    )
    for row in rows:
        row["success_rate"] = success_rate
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No request records were found.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            (*CSV_COLUMNS, "response_text")
            if "response_text" in rows[0]
            else CSV_COLUMNS
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert token-level LLM results to one CSV row per request.")
    parser.add_argument("input_json", type=Path, nargs="?", default=Path("results.json"))
    parser.add_argument("output_csv", type=Path, nargs="?", default=None)
    parser.add_argument("--input-tokens", type=Path, default=None)
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()

    output = args.output_csv
    if output is None:
        output = (
            args.input_json.with_name("output.csv")
            if args.input_json.name == "results.json"
            else args.input_json.with_name(f"{args.input_json.stem}_from_json.csv")
        )
    token_path = args.input_tokens or args.input_json.with_name("input_tokens.json")
    rows = aggregate_requests(load_records(args.input_json), load_input_tokens(token_path), args.include_text)
    write_csv(rows, output)
    print(f"Wrote {len(rows):,} request rows to {output}")


if __name__ == "__main__":
    main()
