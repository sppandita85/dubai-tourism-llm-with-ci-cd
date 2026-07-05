#!/usr/bin/env python
"""Training-phase tests.

The overfit test is the end-to-end proof that the backward pass + optimizer are correct:
a small model trained repeatedly on ONE fixed batch must be able to memorize it, driving
the loss far below the random-init baseline ln(vocab_size). A wrong gradient cannot do this.

    python 07_training/tests/test_training.py
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
from model import GPTModel, cross_entropy   # noqa: E402
from optimizer import AdamW                  # noqa: E402
from dataloader import BatchSampler          # noqa: E402


def main() -> int:
    failures = []
    rng = np.random.default_rng(0)

    # --- AdamW reduces a simple quadratic ---
    p = {"w": np.array([5.0, -3.0])}
    opt = AdamW(lr=0.1)
    for _ in range(200):
        g = {"w": 2.0 * p["w"]}              # grad of w^2
        opt.step(p, g)
    if np.abs(p["w"]).max() > 1e-2:
        failures.append(f"AdamW did not minimize quadratic: {p['w']}")

    # --- Overfit test: memorize one fixed batch ---
    V, E, C, H, L = 40, 32, 16, 2, 2
    model = GPTModel(vocab_size=V, emb_dim=E, context_length=C, num_heads=H,
                     num_layers=L, hidden_mult=2, dropout=0.0, seed=1)
    x = rng.integers(0, V, size=(2, C))
    y = rng.integers(0, V, size=(2, C))
    baseline = cross_entropy(model.forward(x, training=False), y)
    opt = AdamW(lr=3e-3, weight_decay=0.0)
    params = model.parameters()
    losses = []
    for _ in range(150):
        loss = model.loss_and_backward(x, y)
        opt.step(params, model.gradients())
        losses.append(loss)
    final = losses[-1]

    if not math.isclose(baseline, math.log(V), rel_tol=0.2):
        failures.append(f"baseline {baseline:.3f} not near ln(V)={math.log(V):.3f}")
    if final > 0.5:
        failures.append(f"overfit failed: loss only reached {final:.3f} (want < 0.5)")
    if not (losses[0] > losses[-1]):
        failures.append("loss did not decrease")

    # --- Checkpoint round-trip ---
    ckpt = ROOT / "tests" / "_tmp_ckpt.npz"
    model.save(str(ckpt))
    reloaded = GPTModel.load(str(ckpt))
    l1 = cross_entropy(model.forward(x, training=False), y)
    l2 = cross_entropy(reloaded.forward(x, training=False), y)
    if not math.isclose(l1, l2, rel_tol=1e-5):
        failures.append(f"checkpoint reload changed loss: {l1} vs {l2}")
    ckpt.unlink(missing_ok=True)

    if failures:
        print("TRAINING TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"All training tests passed.")
    print(f"  overfit: baseline {baseline:.3f} (ln V={math.log(V):.3f}) -> final {final:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
