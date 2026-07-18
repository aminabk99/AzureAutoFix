"""
AzureAutoFix — Model Inference
Loaded by the FastAPI backend to classify errors.
Place azure_error_model.pt and vocab.json in the model/ directory after training.
"""

import json
import math
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent

# ── Lazy torch ───────────────────────────────────────────────────────────────
# torch is NOT imported at module load. Two reasons:
#
#   * Cold start. `import torch` costs 1-2s on a Railway free-tier container.
#     Tiers 1 and 2 of classify() (curated lookup, then retrieval) answer the
#     overwhelming majority of traffic and need none of it, so paying that cost
#     at import made every deploy slower for no benefit.
#   * Degradation. If torch isn't installed or the checkpoint was never
#     trained, the service should lose the model tier -- not fail to boot.
#
# The architecture below must still match model/train.py.

_arch_cache = None


def _arch():
    """Import torch and build the architecture classes on first use."""
    global _arch_cache
    if _arch_cache is not None:
        return _arch_cache

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
            self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, num_classes)

        def forward(self, x):
            x = self.pos_enc(self.embedding(x))
            for blk in self.blocks:
                x = blk(x)
            return self.classifier(self.norm(x).mean(dim=1))

    _arch_cache = {
        "torch": torch,
        "AzureErrorClassifier": AzureErrorClassifier,
    }
    return _arch_cache


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
    a = _arch()
    torch = a["torch"]
    AzureErrorClassifier = a["AzureErrorClassifier"]
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
    torch = _arch()["torch"]
    tokens = text.lower().split()
    ids = [_vocab.get(t, _vocab["<UNK>"]) for t in tokens]
    ids = ids[:max_len] + [_vocab["<PAD>"]] * max(0, max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def classify(error_input: str) -> dict:
    """
    Resolve a raw error code or free-text description to a remediation.

    Three tiers, cheapest and most certain first:

      1. Exact lookup against the 15 curated errors  -- O(1) dict hit, ~0.02ms,
         confidence 1.0. This is the overwhelming majority of real traffic, and
         it deliberately runs *before* the model is even touched.
      2. Hybrid retrieval over the full ~350-code AADSTS catalog (BM25 +
         char-ngram TF-IDF, see model/retrieval.py) -- ~1ms, cites the
         Microsoft doc it matched.
      3. The from-scratch transformer, as a last resort for phrasing that
         retrieval couldn't place.

    If all three abstain, we say so rather than emit a confident-looking guess.
    Tier 1 and 2 need no model, so a missing/untrained checkpoint degrades
    coverage instead of breaking the endpoint.
    """
    import re
    error_code_match = re.search(r"AADSTS\d+", error_input.upper())
    error_code = error_code_match.group(0) if error_code_match else None

    # ── Tier 1: curated exact lookup ─────────────────────────────────────
    global _error_lookup
    if _error_lookup is None:
        try:
            _error_lookup = _load_error_lookup()
        except Exception:
            _error_lookup = {}

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
            "abstained": False,
            "citations": [{
                "title": f'Microsoft Entra error reference — {entry["error_code"]}',
                "url": ("https://learn.microsoft.com/en-us/entra/identity-platform/"
                        f'reference-error-codes#{entry["error_code"].lower()}'),
                "score": 1.0,
            }],
        }

    # ── Tier 2: retrieval over the full AADSTS catalog ───────────────────
    try:
        from model.retrieval import get_retriever
        hit = get_retriever().classify(error_input)
        if hit is not None:
            return hit
    except FileNotFoundError:
        # Catalog not built yet -- fall through to the model.
        pass
    except Exception as exc:
        print(f"[classify] retrieval unavailable ({type(exc).__name__}: {exc})")

    # ── Tier 3: transformer inference ────────────────────────────────────
    if _model is None:
        try:
            load_model()
        except Exception as exc:
            return _abstain(error_code, f"No model available ({type(exc).__name__}).")

    torch = _arch()["torch"]
    with torch.no_grad():
        logits = _model(_encode(error_input))
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)
        fix_category = _idx2label[str(pred_idx.item())]
        conf = round(confidence.item(), 3)

    # The model was trained on 15 examples across a handful of classes. Below
    # this threshold its output is not worth acting on, so abstain rather than
    # route someone to a Graph API write on a coin flip.
    if conf < MODEL_CONFIDENCE_FLOOR:
        return _abstain(
            error_code,
            f"Model confidence {conf:.0%} is below the {MODEL_CONFIDENCE_FLOOR:.0%} "
            f"floor required to act on a prediction.",
        )

    return {
        "error_code": error_code or "UNKNOWN",
        "fix_category": fix_category,
        "user_or_admin": "admin" if "admin" in fix_category else "user",
        "explanation": f"Inferred from model — {error_input}",
        "reasoning": f"Model classified this as {fix_category} with {conf:.0%} confidence.",
        "action": fix_category,
        "action_detail": "Review the fix category and apply appropriate remediation.",
        "user_message": f"Error detected. Category: {fix_category}.",
        "confidence": conf,
        "source": "model",
        "abstained": False,
        "citations": [],
    }


# Predictions weaker than this are reported as "unknown" instead of acted on.
MODEL_CONFIDENCE_FLOOR = float(os.getenv("MODEL_CONFIDENCE_FLOOR", "0.55"))


def _abstain(error_code: str | None, why: str) -> dict:
    """
    Explicit 'I don't know'. Emitting this is a feature: a wrong auto-fix
    against a live tenant is far more costly than saying nothing.
    """
    return {
        "error_code": error_code or "UNKNOWN",
        "fix_category": "admin_escalate",
        "user_or_admin": "admin",
        "explanation": "This error could not be matched to a known Azure AD error.",
        "reasoning": why,
        "action": "manual_review",
        "action_detail": "Escalate to an administrator for manual review.",
        "user_message": "We couldn't identify this error automatically.",
        "confidence": 0.0,
        "source": "abstain",
        "abstained": True,
        "citations": [],
    }
