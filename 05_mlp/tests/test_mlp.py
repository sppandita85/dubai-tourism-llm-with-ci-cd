#!/usr/bin/env python
"""Tests for the FeedForward (MLP) component.

    python 05_mlp/tests/test_mlp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mlp import FeedForward, gelu, relu  # noqa: E402


def main() -> int:
    failures = []
    D, B, T = 16, 2, 5
    ff = FeedForward(D, hidden_mult=4, activation="gelu", dropout=0.0, seed=0)
    x = np.random.default_rng(0).normal(size=(B, T, D)).astype(np.float32)
    y = ff(x)

    # 1. Shape preserved, hidden = 4*D.
    if y.shape != (B, T, D):
        failures.append(f"shape: expected {(B, T, D)}, got {y.shape}")
    if ff.hidden != 4 * D:
        failures.append(f"hidden: expected {4 * D}, got {ff.hidden}")

    # 2. Position-wise: each position is transformed independently (row i of output
    #    depends only on row i of input). Zeroing one position leaves others unchanged.
    x2 = x.copy()
    x2[:, 0, :] = 0.0
    y2 = ff(x2)
    if not np.allclose(y[:, 1:], y2[:, 1:]):
        failures.append("MLP is not position-wise (changing one position affected others)")

    # 3. Parameter count = 2*D*hidden (+ hidden + D with bias).
    if ff.num_parameters != 2 * D * ff.hidden + ff.hidden + D:
        failures.append(f"param count with bias wrong: {ff.num_parameters}")
    nob = FeedForward(D, bias=False, seed=0)
    if nob.num_parameters != 2 * D * nob.hidden:
        failures.append(f"param count without bias wrong: {nob.num_parameters}")

    # 4. GELU properties: gelu(0)=0, ~identity for large +x, ~0 for large -x.
    if not np.isclose(gelu(np.array([0.0]))[0], 0.0):
        failures.append("gelu(0) != 0")
    if not np.isclose(gelu(np.array([10.0]))[0], 10.0, atol=1e-3):
        failures.append("gelu(large +) not ~identity")
    if abs(gelu(np.array([-10.0]))[0]) > 1e-3:
        failures.append("gelu(large -) not ~0")

    # 5. ReLU option works and clamps negatives.
    r = FeedForward(D, activation="relu", seed=0)
    if r.activation_name != "relu" or relu(np.array([-1.0, 2.0]))[0] != 0.0:
        failures.append("relu activation misbehaves")

    # 6. Guards: unknown activation, wrong last dim.
    try:
        FeedForward(D, activation="swish"); failures.append("no error on unknown activation")
    except ValueError:
        pass
    try:
        ff(np.zeros((B, T, D + 1), dtype=np.float32)); failures.append("no error on bad width")
    except ValueError:
        pass

    if failures:
        print("MLP TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All MLP tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
