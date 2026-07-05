#!/usr/bin/env python
"""Fine-tune (continue training) an existing checkpoint on a batch of data.

    python 09_finetuning/scripts/finetune.py \
        --from checkpoints/v0.1.0/model.npz --batch 2026-08 --version v0.2.0

Loads the checkpoint's model, continues training on the batch's tokens with a lower LR,
saves a new checkpoint, and logs a run with strategy="continued".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # finetuning/
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
sys.path.insert(0, str(REPO / "07_training" / "src"))
from model import GPTModel                       # noqa: E402
from dataloader import load_tokens, train_val_split, BatchSampler  # noqa: E402
from trainer import train, evaluate_loss          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_ckpt", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}", args.batch):
        print(f"Batch id must be YYYY-MM (got: {args.batch!r})", file=sys.stderr)
        return 2

    cfg = yaml.safe_load((ROOT / "config" / "finetune.yaml").read_text())
    steps = args.steps if args.steps is not None else cfg["steps"]

    model = GPTModel.load(str(REPO / args.from_ckpt))
    print(f"Loaded base checkpoint {args.from_ckpt}: {model.num_parameters:,} params")

    ids, vocab_size = load_tokens(args.batch)
    if vocab_size != model.vocab_size:
        print(f"ERROR: batch vocab {vocab_size} != model vocab {model.vocab_size} "
              "(tokenizer mismatch)", file=sys.stderr)
        return 1
    train_ids, val_ids = train_val_split(ids, cfg["val_frac"])

    val_sampler = BatchSampler(val_ids, model.context_length, 1, seed=1)
    before = evaluate_loss(model, val_sampler)
    print(f"Fine-tuning on batch {args.batch} ({len(ids):,} tokens) for {steps} steps, "
          f"lr={cfg['lr']}")
    print(f"  val loss before: {before:.4f}")

    history = train(model, train_ids, val_ids,
                    context_length=model.context_length, batch_size=cfg["batch_size"],
                    steps=steps, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                    eval_every=cfg["eval_every"], seed=cfg["seed"])
    after = evaluate_loss(model, val_sampler)
    print(f"  val loss after : {after:.4f}")

    ckpt_dir = REPO / "checkpoints" / args.version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / "model.npz"))
    print(f"Saved checkpoint -> {ckpt_dir/'model.npz'}")

    run = {
        "run_id": f"{args.version}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
        "trained_at": date.today().isoformat(),
        "model_version": args.version,
        "strategy": "continued",
        "base_checkpoint": args.from_ckpt,
        "batches": [args.batch],
        "checkpoint": str((ckpt_dir / "model.npz").relative_to(REPO)),
        "steps": steps,
        "eval": {"val_loss_before": round(before, 4),
                 "val_loss_after": round(after, 4)},
    }
    manifest = REPO / "01_training_input_data" / "manifests" / "training_runs.jsonl"
    with open(manifest, "a") as f:
        f.write(json.dumps(run) + "\n")
    print(f"Logged run {run['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
