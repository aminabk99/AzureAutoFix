"""
AzureAutoFix — Model Inference
Loaded by the FastAPI backend to classify errors.
Place azure_error_model.pt and vocab.json in the model/ directory after training.
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MODEL_DIR = Path(__file__).parent

# ── Architecture (must match train.py) ──────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        def project(linear):
            return linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        Q, K, V = project(self.q), project(self.k), project(self.v)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.dropout(self.ff(x)))


class AzureErrorClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 ff_dim=256, max_len=64, num_classes=5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        return self.classifier(self.norm(x).mean(dim=1))


# ── Loader ───────────────────────────────────────────────────────────────────

_model = None
_vocab = None
_idx2label = None
_error_lookup = None

def _load_error_lookup():
    lookup_path = MODEL_DIR.parent / "data" / "azure_errors.json"
    with open(lookup_path) as f:
        data = json.load(f)
    return {d["error_code"]: d for d in data}

def load_model():
    global _model, _vocab, _idx2label, _error_lookup
    checkpoint = torch.load(MODEL_DIR / "azure_error_model.pt", map_location="cpu", weights_only=False)
    with open(MODEL_DIR / "vocab.json") as f:
        _vocab = json.load(f)
    _idx2label = checkpoint["idx2label"]
    _error_lookup = _load_error_lookup()
    _model = AzureErrorClassifier(
        vocab_size=checkpoint["vocab_size"],
        num_classes=checkpoint["num_classes"],
    )
    _model.load_state_dict(checkpoint["model_state"])
    _model.eval()
    print("[Model] AzureErrorClassifier loaded.")


def _encode(text, max_len=64):
    tokens = text.lower().split()
    ids = [_vocab.get(t, _vocab["<UNK>"]) for t in tokens]
    ids = ids[:max_len] + [_vocab["<PAD>"]] * max(0, max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def classify(error_input: str) -> dict:
    """
    Takes a raw error code string or description, returns classification dict.
    Tries exact lookup first, then falls back to model inference.
    """
    if _model is None:
        load_model()

    # Extract error code if it looks like AADSTS*
    import re
    code_match = re.search(r"AADSTS\d+", error_input.upper())
    error_code = code_match.group(0) if code_match else None

    # Exact lookup from dataset
    if error_code and _error_lookup and error_code in _error_lookup:
        entry = _error_lookup[error_code]
        return {
            "error_code": entry["error_code"],
            "fix_category": entry["fix_category"],
            "user_or_admin": entry["user_or_admin"],
            "explanation": entry["cause"],
            "reasoning": entry["reasoning"],
            "action": entry["action"],
            "action_detail": entry["action_detail"],
            "user_message": entry["user_message"],
            "confidence": 1.0,
            "source": "lookup",
        }

    # Model inference for unknown / descriptive inputs
    with torch.no_grad():
        logits = _model(_encode(error_input))
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)
        fix_category = _idx2label[str(pred_idx.item())]

    return {
        "error_code": error_code or "UNKNOWN",
        "fix_category": fix_category,
        "user_or_admin": "admin" if "admin" in fix_category else "user",
        "explanation": f"Inferred from model — {error_input}",
        "reasoning": f"Model classified this as {fix_category} with {confidence.item():.0%} confidence.",
        "action": fix_category,
        "action_detail": "Review the fix category and apply appropriate remediation.",
        "user_message": f"Error detected. Category: {fix_category}.",
        "confidence": round(confidence.item(), 3),
        "source": "model",
    }
