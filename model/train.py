"""
AzureAutoFix — From-Scratch Transformer for Azure AD Error Classification
Run this in Google Colab (free GPU) or locally.

After training, the model is saved as model/azure_error_model.pt
The tokenizer vocab is saved as model/vocab.json
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os

# ─────────────────────────────────────────
# 1. LOAD + PREPARE DATA
# ─────────────────────────────────────────

DATA_PATH = "../data/azure_errors.json"

with open(DATA_PATH) as f:
    raw_data = json.load(f)

# Labels: all unique fix categories
FIX_CATEGORIES = sorted(set(d["fix_category"] for d in raw_data))
LABEL2IDX = {label: i for i, label in enumerate(FIX_CATEGORIES)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(FIX_CATEGORIES)

print(f"Fix categories ({NUM_CLASSES}): {FIX_CATEGORIES}")

# Input: error_code + cause (concatenated) → fix_category
def build_text(entry):
    return f"{entry['error_code']} {entry['cause']} {entry['reasoning']}"

texts = [build_text(d) for d in raw_data]
labels = [LABEL2IDX[d["fix_category"]] for d in raw_data]

# ─────────────────────────────────────────
# 2. TOKENIZER (character + word level, simple)
# ─────────────────────────────────────────

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts)
VOCAB_SIZE = len(vocab)
print(f"Vocab size: {VOCAB_SIZE}")

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

def encode(text, vocab, max_len=64):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    # Pad or truncate
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

MAX_LEN = 64

# ─────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────

class ErrorDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.encodings = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

# Augment data: duplicate each sample 5x with slight variation for robust training
augmented_texts, augmented_labels = [], []
for text, label in zip(texts, labels):
    for _ in range(5):
        augmented_texts.append(text)
        augmented_labels.append(label)

dataset = ErrorDataset(augmented_texts, augmented_labels, vocab, MAX_LEN)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ─────────────────────────────────────────
# 4. FROM-SCRATCH TRANSFORMER
# ─────────────────────────────────────────

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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        def project(linear, x):
            return linear(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = project(self.q, x), project(self.k, x), project(self.v, x)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class AzureErrorClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3,
                 ff_dim=256, max_len=64, num_classes=5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Mean pool over sequence (ignoring PAD is handled by near-zero embeddings)
        x = x.mean(dim=1)
        return self.classifier(x)


# ─────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {DEVICE}")

model = AzureErrorClassifier(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    num_heads=4,
    num_layers=3,
    ff_dim=256,
    max_len=MAX_LEN,
    num_classes=NUM_CLASSES,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()

EPOCHS = 80

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(-1) == y_batch).sum().item()
        total += y_batch.size(0)
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.1f}%")

# Save
torch.save({
    "model_state": model.state_dict(),
    "label2idx": LABEL2IDX,
    "idx2label": IDX2LABEL,
    "vocab_size": VOCAB_SIZE,
    "num_classes": NUM_CLASSES,
}, "azure_error_model.pt")

print("Model saved to azure_error_model.pt")
print("Vocab saved to vocab.json")
print(f"Classes: {LABEL2IDX}")
