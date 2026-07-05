"""Data loading for training: token stream -> (input, target) batches.

Reads the token IDs produced by the tokenization phase and yields batches of fixed-length
sequences with their next-token targets (target = input shifted one position left).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_tokens(batch: str) -> tuple[np.ndarray, int]:
    """Load processed/<batch>/tokens.bin as int64 IDs; return (ids, vocab_size)."""
    proc = REPO / "01_training_input_data" / "processed" / batch
    meta = json.loads((proc / "meta.json").read_text())
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    ids = np.fromfile(proc / "tokens.bin", dtype=dtype).astype(np.int64)
    return ids, meta["vocab_size"]


def train_val_split(ids: np.ndarray, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(len(ids) * val_frac))
    return ids[:-n_val], ids[-n_val:]


class BatchSampler:
    """Yields random (x, y) batches of shape (batch_size, context_length)."""

    def __init__(self, ids: np.ndarray, context_length: int, batch_size: int,
                 seed: int | None = None) -> None:
        if len(ids) <= context_length + 1:
            raise ValueError("token stream too short for the requested context_length")
        self.ids = ids
        self.ctx = context_length
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)
        self.max_start = len(ids) - context_length - 1

    def batch(self) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, self.max_start, size=self.bs)
        x = np.stack([self.ids[s:s + self.ctx] for s in starts])
        y = np.stack([self.ids[s + 1:s + 1 + self.ctx] for s in starts])
        return x, y

    def iter_all(self):
        """Deterministic non-overlapping pass (used for validation loss)."""
        for s in range(0, self.max_start, self.ctx):
            x = self.ids[s:s + self.ctx][None, :]
            y = self.ids[s + 1:s + 1 + self.ctx][None, :]
            yield x, y
