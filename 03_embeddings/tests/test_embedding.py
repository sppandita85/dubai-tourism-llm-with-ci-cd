#!/usr/bin/env python
"""Tests for the NumPy InputEmbedding component.

    python 03_embeddings/tests/test_embedding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from input_embedding import InputEmbedding  # noqa: E402


def main() -> int:
    failures = []
    V, D, C = 100, 16, 8  # small toy sizes

    # 1. Output shape is (batch, seq_len, emb_dim).
    emb = InputEmbedding(V, D, C, pos_type="learned", seed=0)
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    out = emb(x)
    if out.shape != (2, 4, D):
        failures.append(f"shape: expected (2,4,{D}), got {out.shape}")
    if out.dtype != np.float32:
        failures.append(f"dtype: expected float32, got {out.dtype}")

    # 2. output == token row + positional row (definition of input embedding).
    manual = emb.tok_emb[1] + emb.pos_emb[0]
    if not np.allclose(out[0, 0], manual):
        failures.append("value: out[0,0] != tok_emb[id] + pos_emb[0]")

    # 3. Positional effect: same id at different positions -> different vectors.
    same = emb(np.array([[7, 7, 7]]))
    if np.allclose(same[0, 0], same[0, 1]):
        failures.append("positional: identical token at pos 0 and 1 gave identical vectors")

    # 4. Learned param count = (V + C) * D; sinusoidal = V * D only.
    if emb.num_parameters != (V + C) * D:
        failures.append(f"learned params: expected {(V+C)*D}, got {emb.num_parameters}")
    sin = InputEmbedding(V, D, C, pos_type="sinusoidal", seed=0)
    if sin.num_parameters != V * D:
        failures.append(f"sinusoidal params: expected {V*D}, got {sin.num_parameters}")

    # 5. Sinusoidal table is deterministic (seed-independent) and bounded in [-1, 1].
    sin2 = InputEmbedding(V, D, C, pos_type="sinusoidal", seed=999)
    if not np.array_equal(sin.pos_emb, sin2.pos_emb):
        failures.append("sinusoidal: table depends on seed (should be fixed)")
    if sin.pos_emb.max() > 1.0 or sin.pos_emb.min() < -1.0:
        failures.append("sinusoidal: values out of [-1, 1]")

    # 6. Guards: seq_len > context_length and out-of-range ids must raise.
    try:
        emb(np.zeros((1, C + 1), dtype=int)); failures.append("no error on seq_len > context")
    except ValueError:
        pass
    try:
        emb(np.array([[V]])); failures.append("no error on out-of-range token id")
    except ValueError:
        pass

    # 7. Sinusoidal requires even emb_dim.
    try:
        InputEmbedding(V, 15, C, pos_type="sinusoidal"); failures.append("no error on odd dim")
    except ValueError:
        pass

    if failures:
        print("EMBEDDING TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All embedding tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
