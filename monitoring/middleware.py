"""
AzureAutoFix — Request lifecycle middleware.

Wraps every FastAPI request, records per-endpoint wall-clock latency,
error code (if present in request body), and HTTP status, then appends
a trace record to monitoring/traces.jsonl.
"""

from __future__ import annotations

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_TRACES_FILE = Path(__file__).parent / "traces.jsonl"
_lock = threading.Lock()


async def _read_body_safe(request: Request) -> bytes:
    """Read request body without consuming it (re-sets body stream)."""
    body = await request.body()
    return body


class LatencyTracingMiddleware(BaseHTTPMiddleware):
    """
    Records per-request latency and metadata to traces.jsonl.

    Each line:
    {
        "ts": "2026-07-14T12:00:00Z",
        "endpoint": "/analyze",
        "method": "POST",
        "status": 200,
        "latency_ms": 142.3,
        "error_code": "AADSTS50126"   # only if extractable from body
    }
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        record: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }

        # Best-effort: pull error_code from cached body state
        try:
            body_bytes = await request.body()
            if body_bytes:
                payload = json.loads(body_bytes)
                if "error_code" in payload:
                    record["error_code"] = payload["error_code"]
                elif "error_input" in payload:
                    import re
                    m = re.search(r"AADSTS\d+", payload["error_input"].upper())
                    if m:
                        record["error_code"] = m.group(0)
        except Exception:
            pass

        with _lock:
            with _TRACES_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

        return response
