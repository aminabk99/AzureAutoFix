"""
AzureAutoFix — Background trace writer.

The previous implementation wrote every trace record to disk synchronously,
inside a threading.Lock, from within an async request handler:

    with _lock:
        with _TRACES_FILE.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

That is blocking file I/O on the event loop. Under concurrency every request
serialises behind every other request's disk write, so the monitoring layer was
itself a meaningful share of the latency it was measuring.

This module replaces it with a bounded, non-blocking queue drained by a single
daemon thread:

  * request handlers call `emit()`, which is a queue put -- O(1), no syscall,
    no disk touch
  * one background thread batches records and flushes them, holding the file
    handle open for its lifetime (reopening per batch made the drain thread
    the bottleneck: ~38% of records dropped at 26k rec/s)
  * the queue is bounded. If the writer falls behind, traces are dropped and
    counted rather than applying backpressure to live requests. Observability
    must never be able to take down the thing it observes.
  * the file is size-capped and rotated, so traces.jsonl cannot grow without
    limit on a long-running deployment

Measured (20k records, 16 threads, see the benchmark in the PR notes):
    realistic  ~1k rec/s   p95  715us -> 126us   0 dropped
    heavy      ~4k rec/s   p95 2335us ->  87us   0 dropped
    saturation ~24k rec/s  p95  11.6ms -> 2.5ms  sheds excess
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from pathlib import Path

_TRACES_FILE = Path(__file__).parent / "traces.jsonl"

# Tunables (env-overridable so small containers can shrink them)
_QUEUE_MAX = int(os.getenv("TRACE_QUEUE_MAX", "10000"))
_FLUSH_INTERVAL_S = float(os.getenv("TRACE_FLUSH_INTERVAL", "0.5"))
_BATCH_MAX = int(os.getenv("TRACE_BATCH_MAX", "256"))
_MAX_BYTES = int(os.getenv("TRACE_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
_KEEP_ROTATIONS = int(os.getenv("TRACE_KEEP_ROTATIONS", "1"))

_queue: "queue.Queue[dict]" = queue.Queue(maxsize=_QUEUE_MAX)
_shutdown_evt = threading.Event()
_thread: threading.Thread | None = None
_thread_lock = threading.Lock()

# Counters are plain ints mutated under the GIL; exact precision isn't required,
# we only need an order-of-magnitude signal.
_stats = {"emitted": 0, "written": 0, "dropped": 0}


def emit(record: dict) -> None:
    """
    Enqueue a trace record. Never blocks, never raises.
    This is the only function request handlers should call.
    """
    _ensure_started()
    _stats["emitted"] += 1
    try:
        _queue.put_nowait(record)
    except queue.Full:
        # Writer is behind. Drop rather than stall the request.
        _stats["dropped"] += 1


def stats() -> dict:
    """Writer health -- surfaced via /metrics so drops are visible."""
    return {**_stats, "queue_depth": _queue.qsize()}


# ── File handling ────────────────────────────────────────────────────────────

_fh = None
_bytes_written = 0


def _open_handle():
    global _fh, _bytes_written
    _close_handle()
    _TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _fh = _TRACES_FILE.open("a", encoding="utf-8")
    try:
        _bytes_written = _TRACES_FILE.stat().st_size
    except OSError:
        _bytes_written = 0
    return _fh


def _close_handle() -> None:
    global _fh
    if _fh is not None:
        try:
            _fh.flush()
            _fh.close()
        except Exception:
            pass
        _fh = None


def _rotate_if_needed() -> None:
    """Size-cap traces.jsonl so it can't grow without bound."""
    global _bytes_written
    if _bytes_written < _MAX_BYTES:
        return
    _close_handle()
    try:
        if _KEEP_ROTATIONS > 0:
            _TRACES_FILE.replace(_TRACES_FILE.with_suffix(".jsonl.1"))
        else:
            _TRACES_FILE.unlink()
    except OSError:
        pass
    _open_handle()


def _drain_batch(batch: list[dict]) -> None:
    global _bytes_written
    if not batch:
        return
    try:
        payload = "".join(json.dumps(r, default=str) + "\n" for r in batch)
        if _fh is None:
            _open_handle()
        _fh.write(payload)
        _fh.flush()
        _bytes_written += len(payload)
        _stats["written"] += len(batch)
        _rotate_if_needed()
    except Exception:
        # Losing traces is acceptable; crashing the writer thread is not.
        _stats["dropped"] += len(batch)


# ── Background drain ─────────────────────────────────────────────────────────

def _run() -> None:
    _open_handle()
    batch: list[dict] = []
    last_flush = time.monotonic()

    while not _shutdown_evt.is_set():
        timeout = max(0.0, _FLUSH_INTERVAL_S - (time.monotonic() - last_flush))
        try:
            batch.append(_queue.get(timeout=timeout or _FLUSH_INTERVAL_S))
        except queue.Empty:
            pass

        # Opportunistically pull everything already queued -- one flush per
        # burst rather than one flush per record.
        while len(batch) < _BATCH_MAX:
            try:
                batch.append(_queue.get_nowait())
            except queue.Empty:
                break

        due = (time.monotonic() - last_flush) >= _FLUSH_INTERVAL_S
        if len(batch) >= _BATCH_MAX or (batch and due):
            _drain_batch(batch)
            batch = []
            last_flush = time.monotonic()

    # Final drain on shutdown so we don't lose the tail.
    while True:
        try:
            batch.append(_queue.get_nowait())
        except queue.Empty:
            break
    _drain_batch(batch)
    _close_handle()


def _ensure_started() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _shutdown_evt.clear()
        _thread = threading.Thread(target=_run, name="trace-writer", daemon=True)
        _thread.start()


def shutdown(timeout: float = 2.0) -> None:
    """Flush pending traces. Called from the FastAPI shutdown hook."""
    global _thread
    if _thread is None or not _thread.is_alive():
        return
    _shutdown_evt.set()
    _thread.join(timeout=timeout)
    _thread = None


atexit.register(shutdown)
