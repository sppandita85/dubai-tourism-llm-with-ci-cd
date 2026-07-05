"""Evaluation metrics: cross-entropy loss and perplexity over a held-out token stream."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
from model import cross_entropy  # noqa: E402


def perplexity(model, ids: np.ndarray, context_length: int, max_windows: int = 200):
    """Mean next-token loss and perplexity over non-overlapping windows of `ids`."""
    losses = []
    step = context_length
    for i, start in enumerate(range(0, len(ids) - context_length - 1, step)):
        if i >= max_windows:
            break
        x = ids[start:start + context_length][None, :]
        y = ids[start + 1:start + 1 + context_length][None, :]
        losses.append(cross_entropy(model.forward(x, training=False), y))
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, float(np.exp(loss))
