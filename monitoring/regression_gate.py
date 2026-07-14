#!/usr/bin/env python3
"""
AzureAutoFix — Classification Regression Gate
==============================================
Tests every known AADSTS error code through classify() and asserts that:
  1. The returned error_code matches the input code exactly.
  2. The source is "lookup" (exact match, confidence 1.0).

All 15 known codes live in data/azure_errors.json and are covered here.
No Azure credentials or network access needed — runs entirely offline.

Exit codes:
  0 = all codes classified correctly (CI green)
  1 = one or more failures (CI red)

Usage:
    python -m monitoring.regression_gate
"""

from __future__ import annotations

import sys

# Import classify — works when run from repo root
from model.inference import classify

# All 15 AADSTS codes defined in data/azure_errors.json
KNOWN_CODES = [
    "AADSTS50126",
    "AADSTS50058",
    "AADSTS65001",
    "AADSTS700016",
    "AADSTS900971",
    "AADSTS50055",
    "AADSTS50057",
    "AADSTS700011",
    "AADSTS90094",
    "AADSTS50076",
    "AADSTS70011",
    "AADSTS50020",
    "AADSTS50053",
    "AADSTS90033",
    "AADSTS700082",
]


def run_regression() -> bool:
    """
    Runs classify() on every known error code.
    Returns True if all pass, False if any fail.
    """
    failures: list[str] = []

    print(f"\n{'Code':<16} {'Returned':<16} {'Source':<10} {'Confidence':<12} {'Status'}")
    print("-" * 66)

    for code in KNOWN_CODES:
        try:
            result = classify(code)
            returned_code = result.get("error_code", "")
            source        = result.get("source", "")
            confidence    = result.get("confidence", 0.0)

            ok = (returned_code == code) and (source == "lookup") and (confidence == 1.0)
            status = "PASS" if ok else "FAIL"
            print(f"{code:<16} {returned_code:<16} {source:<10} {confidence:<12.3f} {status}")

            if not ok:
                failures.append(
                    f"  {code}: got error_code={returned_code!r}, "
                    f"source={source!r}, confidence={confidence}"
                )
        except Exception as exc:
            print(f"{code:<16} {'ERROR':<16} {'—':<10} {'—':<12} FAIL")
            failures.append(f"  {code}: exception — {exc}")

    print()
    if failures:
        print(f"REGRESSION FAILURES ({len(failures)}/{len(KNOWN_CODES)}):")
        for f in failures:
            print(f)
        return False

    print(f"All {len(KNOWN_CODES)}/{len(KNOWN_CODES)} codes classified correctly. Gate passed.")
    return True


if __name__ == "__main__":
    passed = run_regression()
    sys.exit(0 if passed else 1)
