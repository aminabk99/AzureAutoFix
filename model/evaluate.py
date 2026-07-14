"""
AzureAutoFix — Standalone evaluation script
Loads trained weights and reports LOOCV metrics in paper-ready format.

Run from the repo root:
    python model/evaluate.py

Output: confusion matrix + per-class precision/recall/F1 table
(suitable for copy-paste into the README or a paper's results section).

Methodology: Leave-One-Out Cross-Validation — see train_local.py for
the rationale. This script is separate from training so CI can run it
independently to detect metric regressions across commits.
"""

import json
import math
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(REPO_ROOT, "data", "azure_errors.json")
MODEL_PATH = os.path.join(REPO_ROOT, "model", "azure_error_model.pt")
VOCAB_PATH = os.path.join(REPO_ROOT, "model", "vocab.json")

if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
    print("Model weights not found. Run: python model/train_local.py")
    sys.exit(0)  # exit 0 so CI doesn't fail before weights are committed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Load data + vocab ─────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw_data = json.load(f)
with open(VOCAB_PATH) as f:
    vocab = json.load(f)

checkpoint  = torch.load(MODEL_PATH, map_location="cpu")
IDX2LABEL   = checkpoint["idx2label"]
LABEL2IDX   = checkpoint["label2idx"]
NUM_CLASSES = checkpoint["num_classes"]
VOCAB_SIZE  = checkpoint["vocab_size"]
MAX_LEN = 64

FIX_CATEGORIES = sorted(LABEL2IDX.keys())


def tokenize(text): return text.lower().split()

def encode(text):
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokenize(text)]
    return (ids[:MAX_LEN] + [vocab["<PAD>"]] * max(0, MAX_LEN - len(ids)))

def build_text(entry):
    return f"{entry['error_code']} {entry['cause']} {entry['reasoning']}"

texts  = [build_text(d) for d in raw_data]
labels = [LABEL2IDX[d["fix_category"]] for d in raw_data]

# ── Model definition (must match train_local.py / inference.py) ───────────────
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
        self.q   = nn.Linear(d_model, d_model)
        self.k   = nn.Linear(d_model, d_model)
        self.v   = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        def p(l): return l(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        Q, K, V = p(self.q), p(self.k), p(self.v)
        a = F.softmax((Q @ K.transpose(-2,-1)) / math.sqrt(self.d_k), dim=-1)
        return self.out((a @ V).transpose(1,2).contiguous().view(B, T, D))

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
        for b in self.blocks: x = b(x)
        return self.classifier(self.norm(x).mean(dim=1))


class ErrorDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc    = [encode(t) for t in texts]
        self.labels = labels
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.enc[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))

def train_fold(train_texts, train_labels, epochs=80):
    ds     = ErrorDataset(train_texts * 5, train_labels * 5)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model  = AzureErrorClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    crit   = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model

def predict_one(model, text):
    with torch.no_grad():
        return model(torch.tensor([encode(text)], dtype=torch.long)).argmax(-1).item()

# ── LOOCV ─────────────────────────────────────────────────────────────────────
print("Running LOOCV evaluation (15 folds)...")
y_true, y_pred = [], []
for i in range(len(raw_data)):
    tr_t = [t for j, t in enumerate(texts)  if j != i]
    tr_l = [l for j, l in enumerate(labels) if j != i]
    m    = train_fold(tr_t, tr_l)
    p    = predict_one(m, texts[i])
    y_true.append(labels[i])
    y_pred.append(p)

# ── Metrics ───────────────────────────────────────────────────────────────────
tp = {c: 0 for c in range(NUM_CLASSES)}
fp = {c: 0 for c in range(NUM_CLASSES)}
fn = {c: 0 for c in range(NUM_CLASSES)}
for t, p in zip(y_true, y_pred):
    if t == p:  tp[t] += 1
    else:       fp[p] += 1; fn[t] += 1

def prf(c):
    prec = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0.0
    rec  = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    sup  = tp[c]+fn[c]
    return prec, rec, f1, sup

rows = [(IDX2LABEL[str(c)], *prf(c)) for c in range(NUM_CLASSES)]
macro_p = sum(r[1] for r in rows) / NUM_CLASSES
macro_r = sum(r[2] for r in rows) / NUM_CLASSES
macro_f = sum(r[3] for r in rows) / NUM_CLASSES
accuracy = sum(t==p for t,p in zip(y_true,y_pred)) / len(y_true)

# ── Console output ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("LOOCV RESULTS — AzureAutoFix Error Classifier")
print("Vaswani et al. (2017) Transformer, d_model=128, H=4, L=3")
print("=" * 65)
print(f"{'Class':<22} {'Precision':>10} {'Recall':>8} {'F1-score':>9} {'Support':>8}")
print("-" * 65)
for name, prec, rec, f1, sup in rows:
    print(f"{name:<22} {prec:>10.3f} {rec:>8.3f} {f1:>9.3f} {sup:>8}")
print("-" * 65)
print(f"{'macro avg':<22} {macro_p:>10.3f} {macro_r:>8.3f} {macro_f:>9.3f} {len(y_true):>8}")
print()
print(f"LOOCV Accuracy: {accuracy:.3f}  ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")

# ── Confusion matrix ───────────────────────────────────────────────────────────
print()
print("Confusion matrix (rows=true, cols=predicted):")
labels_ordered = [IDX2LABEL[str(i)] for i in range(NUM_CLASSES)]
cm = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

header = f"{'':>20}" + "".join(f"{l[:8]:>10}" for l in labels_ordered)
print(header)
for i, row in enumerate(cm):
    print(f"{labels_ordered[i]:>20}" + "".join(f"{v:>10}" for v in row))

# ── Markdown table (copy-paste into README) ────────────────────────────────────
print()
print("── Markdown table (paste into README) ──")
print()
print("| Class | Precision | Recall | F1-score | Support |")
print("|---|---|---|---|---|")
for name, prec, rec, f1, sup in rows:
    print(f"| {name} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {sup} |")
print(f"| **macro avg** | **{macro_p:.3f}** | **{macro_r:.3f}** | **{macro_f:.3f}** | **{len(y_true)}** |")
print()
print(f"LOOCV accuracy: **{accuracy:.3f}** ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")

# ── CI exit code: fail if macro F1 drops below 0.80 ──────────────────────────
MACRO_F1_THRESHOLD = 0.80
if macro_f < MACRO_F1_THRESHOLD:
    print(f"\nCI FAIL: macro F1 {macro_f:.3f} < threshold {MACRO_F1_THRESHOLD}")
    sys.exit(1)
print(f"\nCI PASS: macro F1 {macro_f:.3f} >= threshold {MACRO_F1_THRESHOLD}")
