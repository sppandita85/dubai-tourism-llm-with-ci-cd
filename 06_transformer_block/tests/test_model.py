#!/usr/bin/env python
"""Tests for LayerNorm, TransformerBlock, and the assembled GPTModel.

    python 06_transformer_block/tests/test_model.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from layer_norm import LayerNorm            # noqa: E402
from transformer_block import TransformerBlock  # noqa: E402
from model import GPTModel, cross_entropy   # noqa: E402


def main() -> int:
    failures = []
    D, H, T, B = 32, 4, 6, 2
    rng = np.random.default_rng(0)

    # --- LayerNorm ---
    ln = LayerNorm(D)
    x = rng.normal(3.0, 5.0, size=(B, T, D)).astype(np.float32)  # shifted/scaled input
    out = ln(x)
    if not np.allclose(out.mean(axis=-1), 0.0, atol=1e-5):
        failures.append("LayerNorm: per-position mean not ~0")
    if not np.allclose(out.std(axis=-1), 1.0, atol=1e-3):
        failures.append("LayerNorm: per-position std not ~1")
    if ln.num_parameters != 2 * D:
        failures.append(f"LayerNorm params: expected {2*D}, got {ln.num_parameters}")

    # --- TransformerBlock ---
    blk = TransformerBlock(D, H, seed=1)
    xb = rng.normal(size=(B, T, D)).astype(np.float32)
    yb = blk(xb)
    if yb.shape != (B, T, D):
        failures.append(f"block shape: expected {(B,T,D)}, got {yb.shape}")
    # Causality must survive the block: change a future token, earlier outputs unchanged.
    xb2 = xb.copy(); xb2[:, T - 1, :] += 5.0
    yb2 = blk(xb2)
    if not np.allclose(yb[:, : T - 1], yb2[:, : T - 1], atol=1e-4):
        failures.append("block breaks causality (future token changed earlier outputs)")
    # Block sums its sublayer param counts.
    if blk.num_parameters != (blk.norm1.num_parameters + blk.attn.num_parameters
                              + blk.norm2.num_parameters + blk.ff.num_parameters):
        failures.append("block param count != sum of sublayers")

    # --- GPTModel ---
    V, L = 50, 3
    model = GPTModel(vocab_size=V, emb_dim=D, context_length=T, num_heads=H,
                     num_layers=L, seed=0)
    ids = rng.integers(0, V, size=(B, T))
    logits = model(ids)
    if logits.shape != (B, T, V):
        failures.append(f"logits shape: expected {(B,T,V)}, got {logits.shape}")
    # Total params = embeddings + L blocks + final norm + output head.
    expected = (model.emb.num_parameters + sum(b.num_parameters for b in model.blocks)
                + model.final_norm.num_parameters + model.out_head.size)
    if model.num_parameters != expected:
        failures.append(f"model param count wrong: {model.num_parameters} vs {expected}")
    if len(model.blocks) != L:
        failures.append(f"expected {L} blocks, got {len(model.blocks)}")

    # Causality end-to-end through the whole model.
    ids2 = ids.copy(); ids2[:, T - 1] = (ids[:, T - 1] + 1) % V
    logits2 = model(ids2)
    if not np.allclose(logits[:, : T - 1], logits2[:, : T - 1], atol=1e-4):
        failures.append("model breaks causality (future token changed earlier logits)")

    # Untrained cross-entropy ~ ln(V) (chance level), and reproducible with same seed.
    loss = cross_entropy(logits, ids)
    if not math.isclose(loss, math.log(V), rel_tol=0.15):
        failures.append(f"random-init loss {loss:.3f} not near ln(V)={math.log(V):.3f}")
    model_b = GPTModel(vocab_size=V, emb_dim=D, context_length=T, num_heads=H,
                       num_layers=L, seed=0)
    if not np.allclose(logits, model_b(ids)):
        failures.append("same seed produced different logits (not reproducible)")

    if failures:
        print("MODEL TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All transformer-block / model tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
