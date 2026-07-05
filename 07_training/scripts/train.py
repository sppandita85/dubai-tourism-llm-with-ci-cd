#!/usr/bin/env python
"""Train the model from scratch on a tokenized batch, save a checkpoint, log the run.

    python 07_training/scripts/train.py 2026-07
    python 07_training/scripts/train.py 2026-07 --steps 50 --version v0.1.0

Saves checkpoints/<version>/model.npz and appends a row to
01_training_input_data/manifests/training_runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # training/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
from model import GPTModel                # noqa: E402
from dataloader import load_tokens, train_val_split  # noqa: E402
from trainer import train, evaluate_loss             # noqa: E402
from dataloader import BatchSampler                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--version", default=None)
    args = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}", args.batch):
        print(f"Batch id must be YYYY-MM (got: {args.batch!r})", file=sys.stderr)
        return 2

    cfg = yaml.safe_load((ROOT / "config" / "train.yaml").read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]
    steps = args.steps if args.steps is not None else tcfg["steps"]
    version = args.version or f"v-{args.batch}-scratch"

    ids, vocab_size = load_tokens(args.batch)
    train_ids, val_ids = train_val_split(ids, tcfg["val_frac"])
    print(f"Training on batch {args.batch}: {len(ids):,} tokens "
          f"({len(train_ids):,} train / {len(val_ids):,} val), vocab={vocab_size}")

    model = GPTModel(vocab_size=vocab_size, seed=tcfg["seed"], **mcfg)
    print(f"Model: {model.num_parameters:,} parameters "
          f"({mcfg['num_layers']} layers, emb_dim {mcfg['emb_dim']})")

    print(f"Training for {steps} steps...")
    history = train(model, train_ids, val_ids,
                    context_length=mcfg["context_length"], batch_size=tcfg["batch_size"],
                    steps=steps, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
                    eval_every=tcfg["eval_every"], seed=tcfg["seed"])

    # Final validation loss.
    val_sampler = BatchSampler(val_ids, mcfg["context_length"], 1, seed=1)
    final_val = evaluate_loss(model, val_sampler)

    ckpt_dir = REPO / "checkpoints" / version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / "model.npz"))
    print(f"Saved checkpoint -> {ckpt_dir/'model.npz'}")

    run = {
        "run_id": f"{version}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
        "trained_at": date.today().isoformat(),
        "model_version": version,
        "strategy": "from_scratch",
        "batches": [args.batch],
        "checkpoint": str((ckpt_dir / "model.npz").relative_to(REPO)),
        "steps": steps,
        "params": model.num_parameters,
        "eval": {"final_train_loss": history[-1]["train_loss"],
                 "final_val_loss": round(final_val, 4)},
    }
    manifest = REPO / "01_training_input_data" / "manifests" / "training_runs.jsonl"
    with open(manifest, "a") as f:
        f.write(json.dumps(run) + "\n")
    print(f"Logged run {run['run_id']}  (final val loss {final_val:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
