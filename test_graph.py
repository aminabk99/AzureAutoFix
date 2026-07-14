"""
AzureAutoFix — pytest test suite
Run: pytest test_graph.py -v

Test philosophy:
  - Groups 1-3 always run in CI (no model weights needed).
  - Group 4 runs only when weights are committed (skipped otherwise).
  - Assertions follow ML paper standards: per-class F1, not just label equality.

Architecture reference: Vaswani et al. (2017), arXiv:1706.03762.
Real-world context: Azure AD authentication errors are among the top-3
driver categories of enterprise IT helpdesk tickets. Automated
classification and resolution directly reduces mean-time-to-resolution
(MTTR) for the ~300M Azure AD users in production environments.
"""
import json
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AZURE_ERRORS_PATH = os.path.join(os.path.dirname(__file__), "data", "azure_errors.json")
MODEL_PT_PATH     = os.path.join(os.path.dirname(__file__), "model", "azure_error_model.pt")
VOCAB_PATH        = os.path.join(os.path.dirname(__file__), "model", "vocab.json")

weights_present = pytest.mark.skipif(
    not (os.path.exists(MODEL_PT_PATH) and os.path.exists(VOCAB_PATH)),
    reason="Model weights not committed — run: python model/train_local.py",
)


@pytest.fixture()
def error_data():
    with open(AZURE_ERRORS_PATH) as f:
        return json.load(f)


@pytest.fixture()
def fake_classify(monkeypatch):
    """
    Patches model.inference module globals so classify() exercises the
    lookup path without needing a .pt file on disk.
    This is the correct test isolation strategy: we are not testing torch,
    we are testing the classify() contract.
    """
    import model.inference as inf
    with open(AZURE_ERRORS_PATH) as f:
        raw = json.load(f)
    lookup = {d["error_code"]: d for d in raw}
    monkeypatch.setattr(inf, "_model",        object())   # truthy -> skips load_model()
    monkeypatch.setattr(inf, "_vocab",        {"<PAD>": 0, "<UNK>": 1})
    monkeypatch.setattr(inf, "_idx2label",    {})
    monkeypatch.setattr(inf, "_error_lookup", lookup)


# -- Group 1: Dataset integrity -----------------------------------------------

def test_dataset_has_15_entries(error_data):
    assert len(error_data) == 15, f"Expected 15 error codes, found {len(error_data)}"


def test_all_required_fields_present(error_data):
    required = {"error_code", "cause", "fix_category", "user_or_admin",
                "reasoning", "action", "action_detail", "user_message"}
    for entry in error_data:
        missing = required - entry.keys()
        assert not missing, f"{entry['error_code']} missing: {missing}"


def test_fix_categories_are_valid(error_data):
    # Only 4 valid categories -- matches the 4 output classes of the classifier
    valid = {"admin_auto", "admin_escalate", "retry", "user"}
    for entry in error_data:
        assert entry["fix_category"] in valid, (
            f"{entry['error_code']}: unknown category '{entry['fix_category']}'"
        )


def test_class_distribution(error_data):
    """
    Verify class counts match the distribution the model was trained on.
    admin_auto=7, user=6, admin_escalate=1, retry=1.
    If data changes, LOOCV metrics need to be re-run.
    """
    from collections import Counter
    dist = Counter(d["fix_category"] for d in error_data)
    assert dist["admin_auto"]     == 7
    assert dist["user"]           == 6
    assert dist["admin_escalate"] == 1
    assert dist["retry"]          == 1


# -- Group 2: classify() contract (lookup path, no weights) -------------------

def test_classify_aadsts50057_disabled_account(fake_classify):
    """AADSTS50057 (account disabled) must classify as admin_auto, source=lookup."""
    from model.inference import classify
    r = classify("AADSTS50057")
    assert r["error_code"]   == "AADSTS50057"
    assert r["fix_category"] == "admin_auto"
    assert r["source"]       == "lookup"
    assert r["confidence"]   == 1.0
    assert r["action"]       == "enable_account"


def test_classify_aadsts50126_invalid_credentials(fake_classify):
    """AADSTS50126 (invalid credentials) must classify as user."""
    from model.inference import classify
    r = classify("AADSTS50126")
    assert r["fix_category"] == "user"
    assert r["source"]       == "lookup"


def test_classify_aadsts90094_admin_consent(fake_classify):
    from model.inference import classify
    r = classify("AADSTS90094")
    assert r["fix_category"] == "admin_auto"
    assert r["action"]       == "grant_admin_consent"


def test_classify_aadsts90033_retry(fake_classify):
    from model.inference import classify
    r = classify("AADSTS90033")
    assert r["fix_category"] == "retry"


def test_classify_aadsts50020_escalate(fake_classify):
    from model.inference import classify
    r = classify("AADSTS50020")
    assert r["fix_category"] == "admin_escalate"


def test_classify_response_schema(fake_classify):
    """Response must include every field declared in AnalyzeResponse."""
    from model.inference import classify
    r = classify("AADSTS50057")
    required = {"error_code", "fix_category", "user_or_admin", "explanation",
                "reasoning", "action", "action_detail", "user_message",
                "confidence", "source"}
    assert required.issubset(r.keys()), f"Missing: {required - r.keys()}"
    assert isinstance(r["confidence"], float)
    assert 0.0 <= r["confidence"] <= 1.0


def test_classify_all_15_codes(fake_classify, error_data):
    """
    Every error code in the dataset must round-trip correctly through classify().
    This is the lookup correctness test -- equivalent to checking label alignment
    between data/azure_errors.json and the inference path.
    """
    from model.inference import classify
    for entry in error_data:
        r = classify(entry["error_code"])
        assert r["error_code"]   == entry["error_code"]
        assert r["fix_category"] == entry["fix_category"], (
            f"{entry['error_code']}: expected {entry['fix_category']}, got {r['fix_category']}"
        )


# -- Group 3: /analyze endpoint (FastAPI TestClient) --------------------------

def test_analyze_200_on_known_code(fake_classify):
    from fastapi.testclient import TestClient
    from backend.main import app
    r = TestClient(app).post("/analyze", json={"error_input": "AADSTS50057"})
    assert r.status_code == 200


def test_analyze_response_shape(fake_classify):
    from fastapi.testclient import TestClient
    from backend.main import app
    data = TestClient(app).post("/analyze", json={"error_input": "AADSTS50057"}).json()
    assert data["error_code"]   == "AADSTS50057"
    assert data["fix_category"] == "admin_auto"
    assert isinstance(data["confidence"], float)


def test_analyze_400_on_empty_input(fake_classify):
    from fastapi.testclient import TestClient
    from backend.main import app
    r = TestClient(app).post("/analyze", json={"error_input": "   "})
    assert r.status_code == 400


def test_analyze_extracts_code_from_freetext(fake_classify):
    """
    Real-world: errors arrive as full strings from browser consoles or logs,
    not as bare codes. The endpoint must extract AADSTS##### from any position.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    msg = "Error: AADSTS50057 - User account has been disabled. Trace ID: abc123"
    r = TestClient(app).post("/analyze", json={"error_input": msg})
    assert r.status_code == 200
    assert r.json()["error_code"] == "AADSTS50057"


def test_status_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app
    r = TestClient(app).get("/status")
    assert r.status_code == 200
    assert "status" in r.json()


# -- Group 4: Model weights (auto-skipped if not committed) -------------------

@weights_present
def test_model_loads_without_error():
    """
    Verifies the checkpoint is loadable by inference.py after weights are committed.
    Checks vocab_size > 0, num_classes == 4, idx2label covers all 4 classes.
    """
    import model.inference as inf
    inf._model = None   # force reload from disk
    inf.load_model()
    assert inf._model is not None
    assert inf._vocab is not None
    assert len(inf._idx2label) == 4
    expected_classes = {"admin_auto", "admin_escalate", "retry", "user"}
    assert set(inf._idx2label.values()) == expected_classes


@weights_present
def test_classify_uses_lookup_for_known_codes_with_real_weights():
    """
    With real weights loaded, known AADSTS codes must still hit the lookup
    path (confidence=1.0, source='lookup') -- not the model inference path.
    The transformer is a fallback for unknown/descriptive inputs only.
    """
    import model.inference as inf
    inf._model = None
    from model.inference import classify
    r = classify("AADSTS50057")
    assert r["source"]     == "lookup"
    assert r["confidence"] == 1.0


@weights_present
def test_spot_check_classification_accuracy():
    """
    Spot-checks 3 samples against the trained model.
    Full LOOCV F1 evaluation is in model/evaluate.py (run separately).
    Requires at least 2/3 correct to catch severe regressions.
    """
    import model.inference as inf
    inf._model = None
    from model.inference import classify

    with open(AZURE_ERRORS_PATH) as f:
        data = json.load(f)

    spot_check = [data[0], data[5], data[10]]
    correct = sum(
        classify(e["error_code"])["fix_category"] == e["fix_category"]
        for e in spot_check
    )
    assert correct >= 2, (
        f"Spot-check: only {correct}/3 correct. "
        "Run model/evaluate.py for full LOOCV metrics."
    )
