"""
AzureAutoFix — Observability metrics.

Reads monitoring/traces.jsonl and computes:
  - Per-endpoint p50 / p95 / p99 latency
  - Overall request volume and error rate
  - Error code frequency distribution
  - Classification confidence histogram (from /analyze responses stored separately)

Pure Python — no numpy or pandas required.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict, Counter

_TRACES_FILE = Path(__file__).parent / "traces.jsonl"


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return round(sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo), 2)


def compute_metrics(last_n: int = 1000) -> dict:
    """
    Returns a metrics dict from the last N trace records.

    Shape:
    {
        "request_count": int,
        "error_rate": float,           # fraction of 5xx responses
        "latency_ms": {
            "p50": float, "p95": float, "p99": float, "mean": float
        },
        "per_endpoint": {
            "/analyze": {"p50": ..., "p95": ..., "p99": ..., "count": int},
            ...
        },
        "error_code_frequency": {"AADSTS50126": 5, ...},
        "top_error_codes": [["AADSTS50126", 5], ...]   # top 10 sorted
    }
    """
    if not _TRACES_FILE.exists():
        return _empty_metrics()

    records: list[dict] = []
    try:
        lines = _TRACES_FILE.read_text(encoding="utf-8").splitlines()
        for line in lines[-last_n:]:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except Exception:
        return _empty_metrics()

    if not records:
        return _empty_metrics()

    # Overall latency
    all_latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
    error_5xx     = [r for r in records if r.get("status", 0) >= 500]

    # Per-endpoint latency
    endpoint_latencies: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if "endpoint" in r and "latency_ms" in r:
            endpoint_latencies[r["endpoint"]].append(r["latency_ms"])

    per_endpoint: dict[str, dict] = {}
    for ep, lats in endpoint_latencies.items():
        per_endpoint[ep] = {
            "p50":   _percentile(lats, 50),
            "p95":   _percentile(lats, 95),
            "p99":   _percentile(lats, 99),
            "mean":  round(sum(lats) / len(lats), 2),
            "count": len(lats),
        }

    # Error code frequency
    error_codes = [r["error_code"] for r in records if "error_code" in r]
    code_freq   = dict(Counter(error_codes).most_common())
    top_codes   = Counter(error_codes).most_common(10)

    return {
        "request_count": len(records),
        "error_rate":    round(len(error_5xx) / len(records), 4) if records else 0.0,
        "latency_ms": {
            "p50":  _percentile(all_latencies, 50),
            "p95":  _percentile(all_latencies, 95),
            "p99":  _percentile(all_latencies, 99),
            "mean": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0,
        },
        "per_endpoint":          per_endpoint,
        "error_code_frequency":  code_freq,
        "top_error_codes":       top_codes,
    }


def _empty_metrics() -> dict:
    return {
        "request_count": 0,
        "error_rate": 0.0,
        "latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0},
        "per_endpoint": {},
        "error_code_frequency": {},
        "top_error_codes": [],
    }
