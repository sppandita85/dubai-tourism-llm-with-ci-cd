#!/usr/bin/env python
"""Evaluation-phase tests.

    python 08_evaluation/tests/test_evaluation.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
from model import GPTModel        # noqa: E402
from evaluate import perplexity    # noqa: E402
from generate import generate      # noqa: E402


def main() -> int:
    failures = []
    V, C = 50, 16
    model = GPTModel(vocab_size=V, emb_dim=32, context_length=C, num_heads=2,
                     num_layers=2, hidden_mult=2, dropout=0.0, seed=0)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, size=400)

    # Random-init perplexity ≈ vocab_size (exp of ~ln V).
    loss, ppl = perplexity(model, ids, C)
    if not math.isclose(ppl, V, rel_tol=0.3):
        failures.append(f"random perplexity {ppl:.1f} not near vocab_size {V}")
    if not math.isclose(ppl, math.exp(loss), rel_tol=1e-6):
        failures.append("perplexity != exp(loss)")

    # Generation: correct length, all ids in range, respects context beyond ctx length.
    out = generate(model, [1, 2, 3], max_new_tokens=40, temperature=0.9, top_k=10, seed=1)
    if len(out) != 3 + 40:
        failures.append(f"generate length {len(out)} != 43")
    if not all(0 <= t < V for t in out):
        failures.append("generated a token id out of vocab range")

    # top_k=1 is greedy/deterministic given a fixed model.
    a = generate(model, [1], max_new_tokens=10, temperature=1.0, top_k=1, seed=1)
    b = generate(model, [1], max_new_tokens=10, temperature=1.0, top_k=1, seed=2)
    if a != b:
        failures.append("top_k=1 (greedy) not deterministic across seeds")

    if failures:
        print("EVALUATION TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All evaluation tests passed.")
    print(f"  random-init perplexity {ppl:.1f} ~ vocab_size {V}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
