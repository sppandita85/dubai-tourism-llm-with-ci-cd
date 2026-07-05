"""Training loop: forward -> loss -> backward -> optimizer step, with checkpointing.

Ties together the model (transformer_block phase), the AdamW optimizer, and the batch
sampler. Used by both the training phase and the fine-tuning phase.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import GPTModel, cross_entropy          # noqa: E402
from optimizer import AdamW                          # noqa: E402
from dataloader import BatchSampler                  # noqa: E402


def evaluate_loss(model: GPTModel, sampler: BatchSampler, max_batches: int = 20) -> float:
    """Mean loss over up to max_batches non-overlapping validation windows."""
    losses = []
    for i, (x, y) in enumerate(sampler.iter_all()):
        if i >= max_batches:
            break
        losses.append(cross_entropy(model.forward(x, training=False), y))
    return float(np.mean(losses)) if losses else float("nan")


def train(model: GPTModel, train_ids: np.ndarray, val_ids: np.ndarray, *,
          context_length: int, batch_size: int, steps: int, lr: float,
          weight_decay: float = 0.0, eval_every: int = 50, seed: int = 0,
          log_fn=print) -> list[dict]:
    """Run `steps` optimizer updates. Returns a history of logged points."""
    opt = AdamW(lr=lr, weight_decay=weight_decay)
    train_sampler = BatchSampler(train_ids, context_length, batch_size, seed=seed)
    val_sampler = BatchSampler(val_ids, context_length, 1, seed=seed + 1)
    params = model.parameters()
    rng = np.random.default_rng(seed)

    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = train_sampler.batch()
        loss = model.loss_and_backward(x, y, rng=rng)
        opt.step(params, model.gradients())

        if step % eval_every == 0 or step == 1 or step == steps:
            val = evaluate_loss(model, val_sampler)
            rec = {"step": step, "train_loss": round(loss, 4),
                   "val_loss": round(val, 4), "secs": round(time.time() - t0, 1)}
            history.append(rec)
            log_fn(f"  step {step:5d}/{steps}  train {loss:7.4f}  val {val:7.4f}"
                   f"  ({rec['secs']}s)")
    return history
