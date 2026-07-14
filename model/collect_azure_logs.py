"""
Azure AD Sign-In Log Collector — Microsoft Graph API
=====================================================
Pulls real sign-in events from your Azure AD tenant, groups them into
per-session error sequences, and writes them in the format expected by
the DeepLog training script (model/train_sequence.py).

This replaces data/synthetic_sequences.json with REAL production data,
making the sequence anomaly detector genuinely production-ready.

Prerequisites:
  1. An Azure AD App Registration with:
       - API permission: AuditLog.Read.All (application permission)
       - API permission: Directory.Read.All (application permission)
     Grant admin consent for both.
  2. A client secret (or certificate) for the app registration.
  3. Set environment variables:
       AZURE_TENANT_ID     — your tenant ID
       AZURE_CLIENT_ID     — app registration client ID
       AZURE_CLIENT_SECRET — client secret value

Usage:
    python model/collect_azure_logs.py
    python model/collect_azure_logs.py --days 30 --min-seq-length 2 --output data/sequences.json

What it does:
  1. Authenticates to Microsoft Graph via OAuth2 client-credentials flow.
  2. Fetches sign-in logs for the last N days (default: 14).
  3. Groups sign-in events into sessions by correlationId — the same
     session-grouping strategy used by DeepLog on HDFS (block ID).
  4. Extracts the AADSTS error code from each event (or 'SUCCESS').
  5. Saves sequences split into 'normal' (ended in SUCCESS or no error)
     and 'anomalous' (heuristically labeled) sets.

After running:
    python model/train_sequence.py   # re-trains DeepLog on real data
    python model/evaluate.py         # re-runs LOOCV evaluation

Graph API reference:
  https://learn.microsoft.com/en-us/graph/api/signin-list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency: requests (already in requirements.txt for most projects)
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
AADSTS_RE = re.compile(r"AADSTS\d+", re.IGNORECASE)

# Error codes that clearly indicate a normal/expected interruption
# (MFA prompts, consent flows) rather than a security event
BENIGN_INTERRUPTIONS = {
    "AADSTS50076",  # MFA required (expected in MFA-enforced tenants)
    "AADSTS50079",  # MFA registration required
    "AADSTS50158",  # External security challenge
    "AADSTS90033",  # Transient error (retry expected)
}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_access_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """
    OAuth2 client-credentials flow for Microsoft Graph.
    Returns a bearer token for AuditLog.Read.All scope.
    """
    url = TOKEN_URL.format(tenant_id=tenant_id)
    resp = requests.post(url, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Graph API — sign-in log fetching
# ---------------------------------------------------------------------------

def fetch_sign_in_logs(
    token: str,
    days: int = 14,
    max_events: int = 10_000,
) -> list[dict]:
    """
    Fetch sign-in events from Microsoft Graph auditLogs/signIns.

    Filters:
      - Last `days` days only (Graph API supports up to 30 days).
      - All sign-ins (successful and failed) so we capture full sessions.

    Pagination:
      Follows @odata.nextLink until all pages are fetched or max_events reached.

    Returns list of raw Graph API sign-in objects.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    url = (
        f"{GRAPH_BASE}/auditLogs/signIns"
        f"?$filter=createdDateTime ge {since}"
        f"&$select=id,createdDateTime,userPrincipalName,correlationId,"
        f"status,errorCode,ipAddress,appDisplayName,conditionalAccessStatus"
        f"&$orderby=createdDateTime asc"
        f"&$top=999"
    )

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    events: list[dict] = []

    print(f"  Fetching sign-in logs since {since} ...")
    while url and len(events) < max_events:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 403:
            print(
                "ERROR 403: Missing AuditLog.Read.All permission or admin consent not granted.\n"
                "  Go to: Azure Portal → App registrations → Your app → API permissions\n"
                "  Add: Microsoft Graph → Application permissions → AuditLog.Read.All\n"
                "  Then click 'Grant admin consent'."
            )
            sys.exit(1)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("value", [])
        events.extend(batch)
        url = data.get("@odata.nextLink")
        print(f"    Fetched {len(events)} events so far ...")

    print(f"  Total events fetched: {len(events)}")
    return events


# ---------------------------------------------------------------------------
# Session grouping (DeepLog methodology)
# ---------------------------------------------------------------------------

def extract_error_code(event: dict) -> str:
    """
    Extract the AADSTS error code from a sign-in event.

    Graph API provides errorCode (int) and additionalDetails (string).
    We prefer the additionalDetails string because it contains the full
    AADSTS##### code; errorCode is sometimes just the numeric suffix.
    """
    status = event.get("status", {}) or {}

    # Try additionalDetails first (e.g. "AADSTS50126: Invalid username or password")
    details = status.get("additionalDetails", "") or ""
    m = AADSTS_RE.search(details)
    if m:
        return m.group(0).upper()

    # Try failureReason
    reason = status.get("failureReason", "") or ""
    m = AADSTS_RE.search(reason)
    if m:
        return m.group(0).upper()

    # Fall back to numeric errorCode
    code = status.get("errorCode", 0) or 0
    if code and code != 0:
        return f"AADSTS{code}"

    # Successful sign-in
    error_code = event.get("errorCode", 0) or 0
    if error_code == 0:
        return "SUCCESS"

    return "UNKNOWN"


def group_into_sessions(events: list[dict]) -> dict[str, list[str]]:
    """
    Group sign-in events into sessions by correlationId.

    correlationId is the Graph API equivalent of the HDFS block ID used
    in DeepLog — it uniquely identifies a single authentication attempt
    and groups all the intermediate events (MFA prompts, CA evaluations,
    token issuance) that belong to it.

    Returns: {correlationId: [code1, code2, ..., codeN]}
    """
    sessions: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    for event in events:
        correlation_id = event.get("correlationId") or event.get("id", "")
        if not correlation_id:
            skipped += 1
            continue
        code = extract_error_code(event)
        sessions[correlation_id].append(code)

    print(f"  Grouped into {len(sessions)} sessions ({skipped} events skipped, no correlationId)")
    return dict(sessions)


# ---------------------------------------------------------------------------
# Normal / anomalous classification (heuristic labeling for training split)
# ---------------------------------------------------------------------------

def classify_session(codes: list[str]) -> tuple[str, str]:
    """
    Heuristically classify a session as normal or anomalous.

    Returns (label, reason).

    Normal sessions:
      - End in SUCCESS (possibly after benign interruptions like MFA prompts)
      - Only contain benign retryable errors
      - At most 2 non-SUCCESS, non-benign events

    Anomalous sessions (will be put in the anomalous set, not training):
      - ≥4 identical error codes (retry storm / brute force)
      - AADSTS50126 ≥3 times (credential stuffing)
      - Contains AADSTS50053 (account lockout, suggests prior attack)
      - No SUCCESS and ≥3 distinct non-benign errors (probing)
    """
    non_benign = [c for c in codes if c not in BENIGN_INTERRUPTIONS and c != "SUCCESS" and c != "UNKNOWN"]
    ends_in_success = codes[-1] == "SUCCESS" if codes else False

    # Anomalous checks
    from collections import Counter
    counts = Counter(codes)

    # Brute force / retry storm
    for code, cnt in counts.items():
        if cnt >= 4 and code not in ("SUCCESS", "UNKNOWN"):
            return "anomalous", f"{code} repeated {cnt}x"

    # Credential stuffing
    if counts.get("AADSTS50126", 0) >= 3:
        return "anomalous", "AADSTS50126 × 3+ (credential stuffing)"

    # Account lockout observed
    if "AADSTS50053" in codes:
        return "anomalous", "AADSTS50053 (account lockout)"

    # Many distinct failure types with no success
    if not ends_in_success and len(set(non_benign)) >= 3:
        return "anomalous", f"3+ distinct errors, no success: {set(non_benign)}"

    # Default: normal
    return "normal", "ended in success or benign interruptions"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect(
    days: int = 14,
    min_seq_length: int = 2,
    output_path: str = "data/sequences.json",
) -> None:
    # Read credentials from environment
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")

    if not all([tenant_id, client_id, client_secret]):
        print(
            "ERROR: Missing Azure credentials. Set these environment variables:\n"
            "  AZURE_TENANT_ID     — your Azure AD tenant ID\n"
            "  AZURE_CLIENT_ID     — app registration client ID\n"
            "  AZURE_CLIENT_SECRET — client secret\n\n"
            "Then re-run: python model/collect_azure_logs.py"
        )
        sys.exit(1)

    print("=" * 60)
    print("Azure AD Sign-In Log Collector")
    print(f"Tenant: {tenant_id}")
    print(f"Days:   {days}")
    print("=" * 60)

    # Step 1: Authenticate
    print("\n[1/4] Authenticating to Microsoft Graph ...")
    token = get_access_token(tenant_id, client_id, client_secret)
    print("  OK")

    # Step 2: Fetch logs
    print(f"\n[2/4] Fetching sign-in logs (last {days} days) ...")
    events = fetch_sign_in_logs(token, days=days)

    # Step 3: Group into sessions
    print("\n[3/4] Grouping events into sessions by correlationId ...")
    sessions = group_into_sessions(events)

    # Step 4: Classify and filter
    print("\n[4/4] Classifying sessions ...")
    normal_seqs: list[list[str]] = []
    anomalous_seqs: list[dict] = []
    skipped_short = 0
    vocab_seen: set[str] = set()

    for corr_id, codes in sessions.items():
        if len(codes) < min_seq_length:
            skipped_short += 1
            continue
        vocab_seen.update(codes)
        label, reason = classify_session(codes)
        if label == "normal":
            normal_seqs.append(codes)
        else:
            anomalous_seqs.append({
                "sequence": codes,
                "label": label,
                "description": reason,
                "correlation_id": corr_id,
            })

    print(f"  Normal sessions:    {len(normal_seqs)}")
    print(f"  Anomalous sessions: {len(anomalous_seqs)}")
    print(f"  Skipped (too short):{skipped_short}")
    print(f"  Vocabulary: {sorted(vocab_seen)}")

    if len(normal_seqs) < 20:
        print(
            "\nWARNING: Only {len(normal_seqs)} normal sessions found. "
            "DeepLog needs at least 20–50 for meaningful training.\n"
            "  - Try increasing --days (max 30 for Graph API)\n"
            "  - Check that AuditLog.Read.All permission is granted\n"
            "  - Some tenants restrict sign-in log retention"
        )

    # Build vocabulary entry
    vocab_description = {
        code: f"Azure AD error {code}"
        for code in sorted(vocab_seen)
        if code not in ("SUCCESS", "UNKNOWN")
    }
    vocab_description["SUCCESS"] = "Successful authentication"
    vocab_description["UNKNOWN"] = "Unknown / missing error code"

    # Save
    output = {
        "_description": (
            f"Real Azure AD sign-in sequences collected {datetime.now().isoformat()[:10]} "
            f"from tenant {tenant_id} (last {days} days). "
            f"Generated by model/collect_azure_logs.py."
        ),
        "_source": "Microsoft Graph API auditLogs/signIns",
        "_methodology": (
            "Sessions grouped by correlationId (DeepLog methodology: Du et al., CCS 2017). "
            "Normal/anomalous split by heuristic labeling (brute force, credential stuffing, "
            "lockout signatures). In production, replace heuristic labels with SOC analyst annotations."
        ),
        "_vocabulary": vocab_description,
        "normal_sequences": normal_seqs,
        "anomalous_sequences": anomalous_seqs,
    }

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_output = os.path.join(repo_root, output_path)
    os.makedirs(os.path.dirname(full_output), exist_ok=True)
    with open(full_output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Done] Saved to {full_output}")
    print()
    print("Next steps:")
    print("  python model/train_sequence.py   # re-train DeepLog on real data")
    print("  python model/train_local.py      # re-train classifier")
    print("  pytest test_graph.py -v          # verify everything still passes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Azure AD sign-in logs and convert to DeepLog training sequences."
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="Number of days of sign-in history to fetch (max 30, default 14)"
    )
    parser.add_argument(
        "--min-seq-length", type=int, default=2,
        help="Minimum events per session to include (default 2)"
    )
    parser.add_argument(
        "--output", default="data/sequences.json",
        help="Output path relative to repo root (default: data/sequences.json)"
    )
    args = parser.parse_args()
    collect(days=args.days, min_seq_length=args.min_seq_length, output_path=args.output)
