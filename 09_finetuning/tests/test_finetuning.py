#!/usr/bin/env python
"""Fine-tuning-phase tests: continuing from a checkpoint updates weights and lowers loss.

    python 09_finetuning/tests/test_finetuning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
sys.path.insert(0, str(REPO / "07_training" / "src"))
from model import GPTModel, cross_entropy   # noqa: E402
from trainer import train                    # noqa: E402


def main() -> int:
    failures = []
    V, C = 40, 16
    rng = np.random.default_rng(0)
    ids = np.tile(rng.integers(0, V, size=64), 8)   # repetitive -> learnable
    train_ids, val_ids = ids[:400], ids[400:]

    # Base model + a short "pre-training".
    base = GPTModel(vocab_size=V, emb_dim=32, context_length=C, num_heads=2,
                    num_layers=2, hidden_mult=2, dropout=0.0, seed=1)
    train(base, train_ids, val_ids, context_length=C, batch_size=8, steps=30,
          lr=3e-3, eval_every=1000, seed=1, log_fn=lambda *_: None)
    ckpt = ROOT / "tests" / "_tmp_base.npz"
    base.save(str(ckpt))

    # Load it and continue (fine-tune).
    model = GPTModel.load(str(ckpt))
    snapshot = {k: v.copy() for k, v in model.parameters().items()}
    x = np.stack([train_ids[i:i + C] for i in range(4)])
    y = np.stack([train_ids[i + 1:i + 1 + C] for i in range(4)])
    before = cross_entropy(model.forward(x, training=False), y)
    train(model, train_ids, val_ids, context_length=C, batch_size=8, steps=60,
          lr=1e-3, eval_every=1000, seed=2, log_fn=lambda *_: None)
    after = cross_entropy(model.forward(x, training=False), y)

    if not (after < before):
        failures.append(f"fine-tuning did not reduce loss: {before:.3f} -> {after:.3f}")
    changed = any(not np.allclose(snapshot[k], v)
                  for k, v in model.parameters().items())
    if not changed:
        failures.append("fine-tuning did not change any weights")

    ckpt.unlink(missing_ok=True)
    if failures:
        print("FINE-TUNING TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All fine-tuning tests passed.")
    print(f"  continued training: loss {before:.3f} -> {after:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
