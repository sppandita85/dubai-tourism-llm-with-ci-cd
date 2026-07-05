#!/usr/bin/env python
"""Tests for CausalMultiHeadAttention.

    python 04_attention/tests/test_attention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from attention import CausalMultiHeadAttention, softmax  # noqa: E402


def main() -> int:
    failures = []
    D, H, T, B = 32, 4, 6, 2
    rng = np.random.default_rng(0)
    attn = CausalMultiHeadAttention(D, H, dropout=0.0, bias=True, seed=1)
    x = rng.normal(size=(B, T, D)).astype(np.float32)
    y = attn(x)

    # 1. Shape preserved.
    if y.shape != (B, T, D):
        failures.append(f"shape: expected {(B, T, D)}, got {y.shape}")

    # 2. Attention weights: rows sum to 1, and no weight on future positions.
    W = attn.last_attn                        # (B, H, T, T)
    if not np.allclose(W.sum(axis=-1), 1.0):
        failures.append("attention rows do not sum to 1")
    future = np.triu(np.ones((T, T), bool), k=1)
    if W[..., future].max(initial=0) > 1e-9:
        failures.append("nonzero attention weight on a future position (mask broken)")

    # 3. Causality end-to-end: perturbing a FUTURE token must not change earlier outputs.
    x2 = x.copy()
    x2[:, T - 1, :] += 5.0                     # change only the last position
    y2 = attn(x2)
    if not np.allclose(y[:, : T - 1], y2[:, : T - 1], atol=1e-5):
        failures.append("output at earlier positions changed when a future token changed")
    if np.allclose(y[:, T - 1], y2[:, T - 1]):
        failures.append("output at the changed position did NOT change (sanity)")

    # 4. num_heads must divide emb_dim.
    try:
        CausalMultiHeadAttention(30, 4); failures.append("no error when heads !| emb_dim")
    except ValueError:
        pass

    # 5. Parameter count = 4*D^2 (+ 4*D with bias); no-bias variant drops the 4*D.
    if attn.num_parameters != 4 * D * D + 4 * D:
        failures.append(f"param count with bias wrong: {attn.num_parameters}")
    nob = CausalMultiHeadAttention(D, H, bias=False, seed=1)
    if nob.num_parameters != 4 * D * D:
        failures.append(f"param count without bias wrong: {nob.num_parameters}")

    # 6. softmax helper: rows sum to 1, handles -inf (masked) entries.
    s = softmax(np.array([[1.0, 2.0, -np.inf]]))
    if not (np.isclose(s.sum(), 1.0) and s[0, 2] == 0.0):
        failures.append(f"softmax with -inf misbehaves: {s}")

    # 7. Input shape guard.
    try:
        attn(np.zeros((B, T, D + 1), dtype=np.float32)); failures.append("no error on bad width")
    except ValueError:
        pass

    if failures:
        print("ATTENTION TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All attention tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
