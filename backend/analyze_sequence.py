"""
/analyze_sequence endpoint — DeepLog-powered sequence anomaly detection
=======================================================================
Adds sequence-level intelligence to AzureAutoFix, going beyond single-error
classification to detect anomalous patterns across a session window.

This implements the second layer of the three-paper AIOps pipeline:
  [1] Drain  → model/parser.py          (parse raw strings -> log keys)
  [2] DeepLog -> THIS FILE              (detect anomalous sequences)
  [3] LogBERT -> model/logbert_classifier.py (classify fix category)

Endpoint: POST /analyze_sequence
Request body:
  {
    "session_errors": ["AADSTS50126", "AADSTS50126", "AADSTS50126"],
    "raw_logs": ["optional list of raw log strings"]
  }

To register this router in backend/main.py, add TWO lines:
    from backend.analyze_sequence import router as sequence_router
    app.include_router(sequence_router)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SequenceRequest(BaseModel):
    session_errors: list[str]
    raw_logs: Optional[list[str]] = None

    @validator("session_errors")
    def validate_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("session_errors must contain at least one error code")
        if len(v) > 50:
            raise ValueError("session_errors must not exceed 50 events")
        return v


class IndividualResult(BaseModel):
    error_code: str
    fix_category: str
    action: str
    confidence: float
    source: str


class SequenceAnalysis(BaseModel):
    is_anomalous: bool
    anomaly_score: float
    anomalous_transitions: list[dict]
    sequence_length: int
    recommendation: str


class SequenceResponse(BaseModel):
    individual_analyses: list[IndividualResult]
    sequence_analysis: SequenceAnalysis
    overall_action: str
    confidence: float
    pipeline: str = "Drain (He et al. ICWS 2017) + DeepLog (Du et al. CCS 2017)"


# ---------------------------------------------------------------------------
# Overall action decision
# ---------------------------------------------------------------------------

def _decide_overall_action(
    individual: list[dict],
    seq_analysis: dict,
) -> tuple[str, float]:
    """
    Combine per-error fix categories with sequence anomaly score.

    Priority (highest first):
      1. High anomaly score (>=0.5) -> security escalation
      2. Any admin_escalate in individuals -> admin_escalate
      3. Moderate anomaly -> review before acting
      4. All retry -> retry
      5. Majority admin_auto -> admin_auto
      6. Majority user -> user
    """
    if seq_analysis["is_anomalous"] and seq_analysis["anomaly_score"] >= 0.5:
        return "escalate_security", 0.9

    categories = [r.get("fix_category", "") for r in individual]

    if "admin_escalate" in categories:
        return "admin_escalate", 0.85

    if seq_analysis["is_anomalous"]:
        return "escalate_review", 0.7

    if all(c == "retry" for c in categories):
        return "retry", 0.95

    if any(c == "admin_auto" for c in categories):
        return "admin_auto", 0.85

    if any(c == "user" for c in categories):
        return "user", 0.85

    return "retry", 0.5


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analyze_sequence", response_model=SequenceResponse)
async def analyze_sequence_endpoint(req: SequenceRequest) -> SequenceResponse:
    """
    Analyze a sequence of Azure AD errors for individual fixes and
    anomalous session-level patterns using the DeepLog pipeline.

    Steps:
      1. Parse each raw error string via Drain log parser (log key extraction).
      2. Classify each log key via the fix-category model.
      3. Run the full key sequence through the DeepLog LSTM anomaly detector.
      4. Combine results into a unified session-level recommendation.
    """
    try:
        from model.inference import classify
        from model.parser import extract_log_key
        from model.sequence_detector import analyze_sequence
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Model import error: {e}")

    # Steps 1-2: Parse and classify individually
    individual_results: list[dict] = []
    for raw in req.session_errors:
        log_key = extract_log_key(raw)
        input_str = log_key if log_key != "UNKNOWN" else raw
        individual_results.append(classify(input_str))

    # Step 3: Sequence anomaly detection
    log_keys = [extract_log_key(e) for e in req.session_errors]
    seq_result = analyze_sequence(log_keys)

    # Step 4: Decide overall action
    overall_action, confidence = _decide_overall_action(individual_results, seq_result)

    return SequenceResponse(
        individual_analyses=[
            IndividualResult(
                error_code=r.get("error_code", "UNKNOWN"),
                fix_category=r.get("fix_category", "unknown"),
                action=r.get("action", ""),
                confidence=r.get("confidence", 0.0),
                source=r.get("source", ""),
            )
            for r in individual_results
        ],
        sequence_analysis=SequenceAnalysis(
            is_anomalous=seq_result["is_anomalous"],
            anomaly_score=seq_result["anomaly_score"],
            anomalous_transitions=seq_result["anomalous_transitions"],
            sequence_length=seq_result["sequence_length"],
            recommendation=seq_result["recommendation"],
        ),
        overall_action=overall_action,
        confidence=confidence,
    )
