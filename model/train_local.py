"""
AzureAutoFix — Local training script
Saves model/azure_error_model.pt and model/vocab.json.

Run from the repo root:
    python model/train_local.py

Architecture: from-scratch Transformer encoder (Vaswani et al., 2017 —
"Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762).
Hyperparameters (d_model=128, H=4, L=3, ff_dim=256) chosen to fit the
dataset scale (N=15) while keeping the checkpoint under 2 MB.

Evaluation: Leave-One-Out Cross-Validation (LOOCV).
LOOCV is the standard methodology for very small datasets (N < 50) —
see Kohavi (1995) "A study of cross-validation and bootstrap for
accuracy estimation and model selection". At N=15 it gives an unbiased
estimate of generalisation error without wasting any samples for a
held-out split.

Reported metrics: precision, recall, F1-score, support — per class and
macro-averaged — matching the reporting standard of NLP classification
papers (e.g., as produced by sklearn.metrics.classification_report).
"""

import json
import math
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "azure_errors.json")
OUT_MODEL = os.path.join(REPO_ROOT, "model", "azure_error_model.pt")
OUT_VOCAB  = os.path.join(REPO_ROOT, "model", "vocab.json")

# ── Data ──────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw_data = json.load(f)

FIX_CATEGORIES = sorted(set(d["fix_category"] for d in raw_data))
LABEL2IDX = {label: i for i, label in enumerate(FIX_CATEGORIES)}
IDX2LABEL  = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(FIX_CATEGORIES)

print(f"Classes ({NUM_CLASSES}): {FIX_CATEGORIES}")
print(f"Dataset size: N={len(raw_data)}")
print(f"Class distribution: { {k: sum(1 for d in raw_data if d['fix_category']==k) for k in FIX_CATEGORIES} }")
print()


def build_text(entry):
    """Concatenate error_code + cause + reasoning as the input sequence."""
    return f"{entry['error_code']} {entry['cause']} {entry['reasoning']}"


texts  = [build_text(d) for d in raw_data]
labels = [LABEL2IDX[d["fix_category"]] for d in raw_data]

# ── Tokeniser ─────────────────────────────────────────────────────────────────
def tokenize(text):
    return text.lower().split()

def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word in counter:
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts)
VOCAB_SIZE = len(vocab)
MAX_LEN = 64

def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids = ids[:MAX_LEN] + [vocab["<PAD>"]] * max(0, MAX_LEN - len(ids))
    return ids

# ── Dataset ───────────────────────────────────────────────────────────────────
class ErrorDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc    = [encode(t) for t in texts]
        self.labels = labels
    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx):
        return (torch.tensor(self.enc[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ── Model (Vaswani et al., 2017) ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.h   = num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        def proj(l): return l(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        Q, K, V = proj(self.q), proj(self.k), proj(self.v)
        attn = F.softmax((Q @ K.transpose(-2,-1)) / math.sqrt(self.d_k), dim=-1)
        return self.out((attn @ V).transpose(1,2).contiguous().view(B, T, D))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ff(x)))

class AzureErrorClassifier(nn.Module):
    """
    Transformer encoder for Azure AD error classification.
    Architecture: embedding → positional encoding → L stacked blocks → mean pool → linear head.
    Reference: Vaswani et al. (2017), arXiv:1706.03762, Section 3.
    """
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 ff_dim=256, max_len=64, num_classes=4, dropout=0.1):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc    = PositionalEncoding(d_model, max_len, dropout)
        self.blocks     = nn.ModuleList([TransformerBlock(d_model, num_heads, ff_dim, dropout)
                                         for _ in range(num_layers)])
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.pos_enc(self.embedding(x))
        for blk in self.blocks:
            x = blk(x)
        return self.classifier(self.norm(x).mean(dim=1))


def make_model():
    return AzureErrorClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)

def train_model(train_texts, train_labels, epochs=80, device="cpu"):
    dataset = ErrorDataset(train_texts * 5, train_labels * 5)  # 5x augmentation
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)
    model   = make_model().to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    crit    = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model

def predict(model, text, device="cpu"):
    enc = torch.tensor([encode(text)], dtype=torch.long).to(device)
    with torch.no_grad():
        return model(enc).argmax(-1).item()

# ── Precision / Recall / F1 helpers ──────────────────────────────────────────
def compute_metrics(y_true, y_pred, num_classes):
    """
    Returns per-class and macro precision, recall, F1, support.
    Follows the reporting format of sklearn.metrics.classification_report.
    """
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    results = {}
    for i in range(num_classes):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        sup  = tp[i] + fn[i]
        results[IDX2LABEL[i]] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
    # macro average
    macro_p = sum(v["precision"] for v in results.values()) / num_classes
    macro_r = sum(v["recall"]    for v in results.values()) / num_classes
    macro_f = sum(v["f1"]        for v in results.values()) / num_classes
    results["macro avg"] = {"precision": macro_p, "recall": macro_r,
                             "f1": macro_f, "support": len(y_true)}
    return results


# ── Leave-One-Out Cross-Validation ────────────────────────────────────────────
# Rationale: LOOCV is the recommended evaluation strategy for N < 50.
# It provides a near-unbiased estimate of generalisation error (Kohavi, 1995).
# With N=15 we cannot afford a held-out split without severely shrinking the
# training set for minority classes (admin_escalate: N=1, retry: N=1).
print("=" * 60)
print("LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
print("Kohavi (1995) — recommended for N < 50")
print("=" * 60)

DEVICE = "cpu"  # LOOCV runs 15 training rounds; CPU is fine
loocv_true, loocv_pred = [], []

for held_out_idx in range(len(raw_data)):
    train_t = [t for i, t in enumerate(texts)  if i != held_out_idx]
    train_l = [l for i, l in enumerate(labels) if i != held_out_idx]
    fold_model = train_model(train_t, train_l, epochs=80, device=DEVICE)
    pred = predict(fold_model, texts[held_out_idx], device=DEVICE)
    loocv_true.append(labels[held_out_idx])
    loocv_pred.append(pred)
    status = "✓" if pred == labels[held_out_idx] else "✗"
    print(f"  Fold {held_out_idx+1:2d}/15 | {raw_data[held_out_idx]['error_code']:<14} "
          f"true={IDX2LABEL[labels[held_out_idx]]:<18} pred={IDX2LABEL[pred]:<18} {status}")

print()
metrics = compute_metrics(loocv_true, loocv_pred, NUM_CLASSES)
correct = sum(t == p for t, p in zip(loocv_true, loocv_pred))
print(f"LOOCV Accuracy: {correct}/{len(raw_data)} = {correct/len(raw_data)*100:.1f}%")
print()
print(f"{'Class':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print("-" * 62)
for cls, m in metrics.items():
    print(f"{cls:<22} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>9}")

# ── Final model: train on all 15 samples ─────────────────────────────────────
print()
print("Training final model on full dataset (N=15)...")
final_model = train_model(texts, labels, epochs=80, device=DEVICE)
print("Done.")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save({
    "model_state": final_model.state_dict(),
    "label2idx":   LABEL2IDX,
    "idx2label":   IDX2LABEL,
    "vocab_size":  VOCAB_SIZE,
    "num_classes": NUM_CLASSES,
}, OUT_MODEL)
with open(OUT_VOCAB, "w") as f:
    json.dump(vocab, f)

print(f"\nSaved: {OUT_MODEL}")
print(f"Saved: {OUT_VOCAB}")
print("\nNext:")
print("  git add model/azure_error_model.pt model/vocab.json")
print("  git commit -m 'Add trained model weights (Transformer, LOOCV eval)'")
