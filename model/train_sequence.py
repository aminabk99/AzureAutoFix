"""
DeepLog LSTM Training Script — Azure AD Sequence Anomaly Detector
=================================================================
Trains the DeepLog LSTM (model/sequence_detector.py) on synthetic Azure AD
authentication session sequences (data/synthetic_sequences.json).

Usage:
    python model/train_sequence.py

Output:
    model/sequence_model.pt   — trained LSTM weights
    model/sequence_vocab.json — log-key vocabulary

Methodology (Du et al., CCS 2017, Section 3.2):
  - Train ONLY on normal sequences (self-supervised, no anomaly labels needed).
  - Sliding window: for each normal sequence S = [k1, k2, ..., kT],
    generate (window, next_key) pairs:
        ([<PAD>, <PAD>, k1, k2, k3], k4)  — window_size=5
        ([<PAD>, k1, k2, k3, k4], k5)
        ...
  - Loss: cross-entropy between predicted and actual next log key.
  - Anomaly detection is fully unsupervised at inference time.

Reference: Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). DeepLog: Anomaly
Detection and Diagnosis from System Logs through Deep Learning. CCS 2017.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "synthetic_sequences.json")
SEQ_MODEL_PATH = os.path.join(REPO_ROOT, "model", "sequence_model.pt")
SEQ_VOCAB_PATH = os.path.join(REPO_ROOT, "model", "sequence_vocab.json")

# ---------------------------------------------------------------------------
# Import model architecture
# ---------------------------------------------------------------------------
sys.path.insert(0, REPO_ROOT)
from model.sequence_detector import DeepLogLSTM, WINDOW_SIZE, HIDDEN_SIZE, NUM_LAYERS

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
SEED = 42


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """
    Build log-key vocabulary from normal training sequences.
    Special tokens: <PAD>=0, <UNK>=1 (following LogBERT convention).
    """
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    counts: Counter = Counter()
    for seq in sequences:
        counts.update(seq)
    for key in sorted(counts):
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Dataset construction (DeepLog sliding window)
# ---------------------------------------------------------------------------

def make_windows(
    sequences: list[list[str]],
    vocab: dict[str, int],
    window_size: int = WINDOW_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (input_window, next_key) training pairs from all normal sequences.

    For a sequence [k1, k2, k3, k4, k5] with window_size=3:
      (PAD, PAD, k1) → k2
      (PAD, k1, k2)  → k3
      (k1, k2, k3)   → k4
      (k2, k3, k4)   → k5

    This is exactly the DeepLog training objective (Section 3.2, Figure 3).
    """
    unk = vocab.get("<UNK>", 1)
    pad = vocab.get("<PAD>", 0)
    X, Y = [], []

    for seq in sequences:
        if len(seq) < 2:
            continue
        indices = [vocab.get(k, unk) for k in seq]
        for i in range(1, len(indices)):
            # Window of up to window_size preceding events, left-padded
            start = max(0, i - window_size)
            window = indices[start:i]
            pad_len = window_size - len(window)
            padded = [pad] * pad_len + window
            X.append(padded)
            Y.append(indices[i])

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: DeepLogLSTM,
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses: list[float] = []
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Evaluation: top-g accuracy on training set (proxy for convergence)
# ---------------------------------------------------------------------------

def evaluate_top_g(
    model: DeepLogLSTM,
    X: torch.Tensor,
    Y: torch.Tensor,
    top_g: int = 3,
) -> float:
    """
    Fraction of held-out next-keys in top-g predictions.
    DeepLog reports near-100% on training data; the interesting metric
    is performance on anomalous sequences at inference time.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)                     # (N, V)
        top_preds = logits.topk(top_g, dim=-1).indices  # (N, g)
        correct = (top_preds == Y.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    print("=" * 60)
    print("DeepLog LSTM Training — AzureAutoFix Sequence Detector")
    print("=" * 60)

    # Load data
    print(f"\n[1/4] Loading sequences from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    normal_seqs: list[list[str]] = data["normal_sequences"]
    print(f"      Normal sequences: {len(normal_seqs)}")
    print(f"      Training on NORMAL sequences only (DeepLog methodology)")

    # Vocabulary
    print("\n[2/4] Building vocabulary")
    vocab = build_vocab(normal_seqs)
    print(f"      Vocabulary size: {len(vocab)} tokens")
    print(f"      Keys: {[k for k in vocab if not k.startswith('<')]}")

    # Windows
    print(f"\n[3/4] Constructing sliding windows (window_size={WINDOW_SIZE})")
    X, Y = make_windows(normal_seqs, vocab, window_size=WINDOW_SIZE)
    print(f"      Training pairs: {len(X)}")

    # Model
    model = DeepLogLSTM(
        vocab_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n      Model: DeepLog LSTM")
    print(f"      Params: {total_params:,}")
    print(f"      Arch:   {NUM_LAYERS}-layer LSTM, hidden={HIDDEN_SIZE}")

    # Train
    print(f"\n[4/4] Training ({EPOCHS} epochs, lr={LR})")
    losses = train(model, X, Y)

    top_g_acc = evaluate_top_g(model, X, Y, top_g=3)
    print(f"\n      Top-3 accuracy (training set): {top_g_acc:.1%}")
    print(f"      Final loss: {losses[-1]:.4f}")

    # Save
    os.makedirs(os.path.dirname(SEQ_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), SEQ_MODEL_PATH)
    with open(SEQ_VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\n[Done] Saved model:  {SEQ_MODEL_PATH}")
    print(f"       Saved vocab:   {SEQ_VOCAB_PATH}")
    print()
    print("Next steps:")
    print("  1. python model/train_local.py    # train fix-category classifier")
    print("  2. pytest test_graph.py -v        # run full test suite")
    print("  3. python model/evaluate.py       # LOOCV evaluation")


if __name__ == "__main__":
    main()
