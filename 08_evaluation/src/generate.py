"""Autoregressive text generation from a trained model.

Repeatedly: feed the last `context_length` tokens, read the next-token logits, sample a
token (temperature + optional top-k), append it, repeat.
"""
from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def generate(model, prompt_ids, max_new_tokens: int = 50, temperature: float = 1.0,
             top_k: int | None = None, seed: int | None = None) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = [int(t) for t in prompt_ids]
    ctx = model.context_length
    for _ in range(max_new_tokens):
        window = np.array([ids[-ctx:]], dtype=np.int64)
        logits = model.forward(window, training=False)[0, -1].astype(np.float64)
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.size:
            cut = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < cut, -np.inf, logits)
        probs = _softmax(logits)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return ids
