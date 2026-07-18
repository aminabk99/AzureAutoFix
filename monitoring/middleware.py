"""
AzureAutoFix — Request lifecycle middleware.

Records per-endpoint wall-clock latency, HTTP status, and (when the handler
supplies one) the AADSTS error code, then hands the record to the background
writer in monitoring/writer.py.

Two things changed from the original implementation, both correctness issues
rather than style:

1. Trace records are no longer written to disk inline. The original held a
   global threading.Lock and did a synchronous file append from inside an
   async handler, so under concurrency every request serialised behind every
   other request's disk write. Records are now enqueued (non-blocking) and
   flushed by a background thread. See writer.py.

2. The middleware no longer reads the request body. The original did:

       response = await call_next(request)
       body_bytes = await request.body()      # after the handler consumed it

   Under Starlette's BaseHTTPMiddleware the receive stream has already been
   drained by the endpoint at that point, so this either returned empty or
   re-buffered the payload purely to re-parse JSON the handler had just
   parsed. Reading it *before* call_next is worse -- it swallows the stream
   and the endpoint receives an empty body unless the channel is replayed.

   Instead the handler annotates `request.state.error_code`, and the
   middleware reads that after the fact. No double parse, no stream games.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from monitoring.writer import emit

# Paths that shouldn't pollute latency percentiles with their own traffic.
_EXCLUDED_PATHS = {"/metrics", "/status", "/favicon.ico"}


class LatencyTracingMiddleware(BaseHTTPMiddleware):
    """
    Emits one trace record per request:

    {
        "ts": "2026-07-18T12:00:00+00:00",
        "endpoint": "/analyze",
        "method": "POST",
        "status": 200,
        "latency_ms": 3.1,
        "error_code": "AADSTS50126",   # if the handler set request.state
        "source": "lookup",            # lookup | retrieval | model
        "confidence": 1.0
    }
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Still record the failure before letting it propagate.
            emit({
                "ts": datetime.now(timezone.utc).isoformat(),
                "endpoint": request.url.path,
                "method": request.method,
                "status": 500,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            })
            raise

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        record: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }

        # Handler-supplied annotations (set in backend/main.py).
        for field in ("error_code", "source", "confidence"):
            value = getattr(request.state, field, None)
            if value is not None:
                record[field] = value

        emit(record)

        # Handy for the demo: prove the latency claim in the response headers.
        response.headers["X-Response-Time-ms"] = str(latency_ms)
        return response
