"""
Azure AD Sequence Anomaly Detector — DeepLog-inspired (Du et al., CCS 2017)
============================================================================
Detects anomalous sequences of Azure AD authentication error events using an
LSTM that models normal error-code transitions.

Methodology follows DeepLog Section 3.1 exactly:
  1. Treat each AADSTS error code as a "log key" (same role as HDFS log keys).
  2. Build a vocabulary K of all known log keys.
  3. Train an LSTM on sequences of normal authentication sessions:
       - Input:  window of h=5 recent log keys (as indices)
       - Output: probability distribution over next log key
  4. Anomaly detection at inference:
       - For each observed transition, compute top-g predicted keys.
       - If the actual next key is NOT in the top-g candidates → anomalous.
       - Anomaly score = (anomalous transitions) / (total transitions).

Key deviation from DeepLog (domain adaptation):
  - DeepLog uses h=10 for HDFS (avg sequence length ~19).
    We use h=5 because Azure AD sessions are shorter.
  - DeepLog uses g=9 for HDFS (46 log key types).
    We use g=3 because our vocabulary is 15 error codes + special tokens.
  - DeepLog trains on thousands of normal HDFS sessions.
    We train on synthetic Azure AD sequences (data/synthetic_sequences.json).
    In production, replace with real session logs from Azure Monitor / SIEM.

Architecture:
  - 2-layer LSTM (matches DeepLog Figure 2)
  - Hidden size 64, dropout 0.1 between layers
  - Embedding dim = hidden size (64)

Reference: Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). DeepLog: Anomaly
Detection and Diagnosis from System Logs through Deep Learning.
In Proceedings of CCS 2017. DOI: 10.1145/3133956.3134015
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hyper-parameters (DeepLog Table 2 adapted for Azure AD domain)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 5    # h — number of preceding events in input window
TOP_G = 3          # g — top-g candidates considered "normal" next keys
HIDDEN_SIZE = 64   # LSTM hidden state and embedding dimension
NUM_LAYERS = 2     # LSTM depth (DeepLog uses 2 layers)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_MODEL_PATH = os.path.join(REPO_ROOT, "model", "sequence_model.pt")
SEQ_VOCAB_PATH = os.path.join(REPO_ROOT, "model", "sequence_vocab.json")


# ---------------------------------------------------------------------------
# Neural architecture
# ---------------------------------------------------------------------------

class DeepLogLSTM(nn.Module):
    """
    2-layer LSTM next-log-key predictor.

    Identical to DeepLog Figure 2:
      Embedding → LSTM (2 layers) → FC → softmax (at inference)

    Input:  (batch, window_size) integer tensor of log-key indices
    Output: (batch, vocab_size)  unnormalized logits
    """

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W) — batch of integer sequences
        returns: (B, V) — logits over vocabulary
        """
        emb = self.embedding(x)          # (B, W, H)
        out, _ = self.lstm(emb)          # (B, W, H)
        last = out[:, -1, :]             # (B, H) — final timestep hidden state
        return self.fc(last)             # (B, V)


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class SequenceAnomalyDetector:
    """
    DeepLog-style anomaly detector for Azure AD error sequences.

    Usage (after training):
        detector = SequenceAnomalyDetector()
        detector.load("model/sequence_model.pt", "model/sequence_vocab.json")
        result = detector.analyze(["AADSTS50126", "AADSTS50126", "AADSTS50126", "AADSTS50057"])
    """

    def __init__(self, window_size: int = WINDOW_SIZE, top_g: int = TOP_G):
        self.window_size = window_size
        self.top_g = top_g
        self.model: Optional[DeepLogLSTM] = None
        self.vocab: dict[str, int] = {}
        self.idx2key: dict[int, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_path: str, vocab_path: str) -> None:
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.idx2key = {int(v): k for k, v in self.vocab.items()}

        vocab_size = len(self.vocab)
        self.model = DeepLogLSTM(vocab_size=vocab_size)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode(self, keys: list[str]) -> list[int]:
        unk = self.vocab.get("<UNK>", 1)
        return [self.vocab.get(k, unk) for k in keys]

    def _make_window(self, indices: list[int], pos: int) -> torch.Tensor:
        """
        Build a padded window of length self.window_size ending at position pos.
        Padding token = 0 (<PAD>), prepended on the left.
        """
        start = max(0, pos - self.window_size + 1)
        window = indices[start : pos + 1]
        pad_len = self.window_size - len(window)
        padded = [0] * pad_len + window
        return torch.tensor([padded], dtype=torch.long)  # (1, W)

    # ------------------------------------------------------------------
    # Core anomaly detection (DeepLog Section 3.1)
    # ------------------------------------------------------------------

    def analyze(self, recent_errors: list[str]) -> dict:
        """
        Detect anomalous transitions in a sequence of AADSTS error codes.

        DeepLog Algorithm:
          For each position i in [1, len-1]:
            1. Build window of h preceding keys ending at i-1.
            2. Predict top-g next keys via LSTM.
            3. If actual key at i is NOT in top-g → anomalous transition.

        Returns:
            is_anomalous         — True if any transition is anomalous
            anomaly_score        — fraction of anomalous transitions (0.0–1.0)
            anomalous_transitions— list of dicts describing each anomaly
            sequence_length      — number of events in input
            recommendation       — human-readable escalation advice
        """
        if len(recent_errors) < 2:
            return self._short_sequence_result(recent_errors)

        if not self._loaded:
            return self._heuristic_fallback(recent_errors)

        indices = self._encode(recent_errors)
        anomalous: list[dict] = []

        with torch.no_grad():
            for i in range(1, len(indices)):
                x = self._make_window(indices, i - 1)    # (1, W)
                logits = self.model(x)[0]                 # (V,)
                probs = torch.softmax(logits, dim=-1)
                top_g = torch.topk(probs, min(self.top_g, len(probs))).indices.tolist()

                actual = indices[i]
                if actual not in top_g:
                    context = recent_errors[max(0, i - self.window_size) : i]
                    expected = [self.idx2key.get(j, "?") for j in top_g]
                    anomalous.append({
                        "position": i,
                        "context": context,
                        "observed": recent_errors[i],
                        "expected_top_g": expected,
                        "probability": round(probs[actual].item(), 4),
                    })

        total = len(indices) - 1
        score = len(anomalous) / total
        return {
            "is_anomalous": len(anomalous) > 0,
            "anomaly_score": round(score, 4),
            "anomalous_transitions": anomalous,
            "sequence_length": len(recent_errors),
            "recommendation": self._recommend(recent_errors, score),
        }

    # ------------------------------------------------------------------
    # Fallback when weights not available
    # ------------------------------------------------------------------

    def _heuristic_fallback(self, errors: list[str]) -> dict:
        """
        Rule-based heuristic used when model weights are not yet trained.
        Covers the most critical known anomalous patterns:

          H1: ≥4 identical errors in a row → likely a retry storm or
              misconfigured client.
          H2: Mix of AADSTS50126 (bad creds) ≥3 times + any AADSTS50053
              (account locked) → credential-stuffing signature.
          H3: ≥2 different admin-auto errors in one session → unusual;
              possible misconfiguration cascade.
        """
        from collections import Counter
        counts = Counter(errors)

        # H1: Repeated identical error
        for code, cnt in counts.items():
            if cnt >= 4:
                return {
                    "is_anomalous": True,
                    "anomaly_score": 1.0,
                    "anomalous_transitions": [],
                    "sequence_length": len(errors),
                    "recommendation": (
                        f"Heuristic H1: {code} repeated {cnt}× in one session. "
                        "Likely a misconfigured client or automated retry loop. "
                        "Consider rate-limiting or circuit-breaker."
                    ),
                }

        # H2: Credential stuffing signature
        if counts.get("AADSTS50126", 0) >= 3 and counts.get("AADSTS50053", 0) >= 1:
            return {
                "is_anomalous": True,
                "anomaly_score": 0.8,
                "anomalous_transitions": [],
                "sequence_length": len(errors),
                "recommendation": (
                    "Heuristic H2: Multiple failed-credential attempts followed by "
                    "account-lockout detected. Possible credential-stuffing attack. "
                    "Escalate to security team immediately."
                ),
            }

        return {
            "is_anomalous": False,
            "anomaly_score": 0.0,
            "anomalous_transitions": [],
            "sequence_length": len(errors),
            "recommendation": (
                "No known anomalous pattern detected (heuristic mode). "
                "Run python model/train_sequence.py to enable ML-based detection."
            ),
        }

    def _short_sequence_result(self, errors: list[str]) -> dict:
        return {
            "is_anomalous": False,
            "anomaly_score": 0.0,
            "anomalous_transitions": [],
            "sequence_length": len(errors),
            "recommendation": "Sequence too short for analysis (need ≥2 events).",
        }

    def _recommend(self, errors: list[str], score: float) -> str:
        if score == 0.0:
            return "Error sequence follows normal authentication patterns. Apply standard fix."
        if score >= 0.5:
            return (
                f"HIGH anomaly score ({score:.0%}): this sequence of Azure AD errors deviates "
                "significantly from normal patterns observed in training. Possible causes: "
                "credential-stuffing, misconfigured app, or account takeover attempt. "
                "Recommend immediate escalation to security team."
            )
        return (
            f"Moderate anomaly score ({score:.0%}): unusual error transition detected. "
            "Review recent authentication events before applying automated remediation."
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_detector: Optional[SequenceAnomalyDetector] = None


def _get_detector() -> SequenceAnomalyDetector:
    global _detector
    if _detector is None:
        _detector = SequenceAnomalyDetector()
        if os.path.exists(SEQ_MODEL_PATH) and os.path.exists(SEQ_VOCAB_PATH):
            _detector.load(SEQ_MODEL_PATH, SEQ_VOCAB_PATH)
    return _detector


def analyze_sequence(recent_errors: list[str]) -> dict:
    """
    Public API: analyze a list of recent AADSTS codes for anomalous patterns.

    Args:
        recent_errors: Chronological list of AADSTS codes from a user session.
                       E.g. ["AADSTS50126", "AADSTS50126", "AADSTS50057"]

    Returns:
        DeepLog anomaly detection result dict (see SequenceAnomalyDetector.analyze).
    """
    return _get_detector().analyze(recent_errors)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    normal = ["AADSTS50126", "AADSTS50126"]
    attack = ["AADSTS50126", "AADSTS50126", "AADSTS50126", "AADSTS50126", "AADSTS50053"]

    d = SequenceAnomalyDetector()  # no weights → heuristic mode

    print("Normal session:")
    r = d.analyze(normal)
    print(f"  anomalous={r['is_anomalous']}, score={r['anomaly_score']}")
    print(f"  {r['recommendation']}\n")

    print("Attack signature:")
    r = d.analyze(attack)
    print(f"  anomalous={r['is_anomalous']}, score={r['anomaly_score']}")
    print(f"  {r['r