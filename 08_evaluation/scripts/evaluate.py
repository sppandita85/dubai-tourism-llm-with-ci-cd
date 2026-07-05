#!/usr/bin/env python
"""Evaluate a trained checkpoint: perplexity on held-out tokens + a text sample.

    python 08_evaluation/scripts/evaluate.py --checkpoint checkpoints/v0.1.0/model.npz --batch 2026-07

Uses the held-out tail of the batch's tokens (or the dedicated holdout set if tokenized).
Appends an evaluation record to 01_training_input_data/manifests/training_runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent      # evaluation/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
sys.path.insert(0, str(REPO / "07_training" / "src"))
from model import GPTModel                       # noqa: E402
from dataloader import load_tokens, train_val_split  # noqa: E402
from evaluate import perplexity                   # noqa: E402
from generate import generate                     # noqa: E402


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(path))
    except Exception as e:                          # noqa: BLE001
        print(f"(tokenizer unavailable, showing token ids: {e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "eval.yaml").read_text())
    model = GPTModel.load(str(REPO / args.checkpoint))
    print(f"Loaded {args.checkpoint}: {model.num_parameters:,} params, "
          f"context_length={model.context_length}")

    ids, _ = load_tokens(args.batch)
    _, val_ids = train_val_split(ids, 0.1)         # held-out tail
    loss, ppl = perplexity(model, val_ids, model.context_length, cfg["max_windows"])
    print(f"\nHeld-out evaluation (batch {args.batch}, {len(val_ids):,} tokens):")
    print(f"  cross-entropy loss : {loss:.4f}")
    print(f"  perplexity         : {ppl:.2f}   (lower is better; random ≈ vocab_size)")

    # ---- qualitative text sample ----
    gcfg = cfg["generate"]
    tok = load_tokenizer(REPO / cfg["tokenizer_json"])
    if tok is not None and gcfg["prompt"]:
        prompt_ids = tok.encode(gcfg["prompt"]).ids
    else:
        prompt_ids = [101]                         # [CLS]
    out_ids = generate(model, prompt_ids, max_new_tokens=gcfg["max_new_tokens"],
                       temperature=gcfg["temperature"], top_k=gcfg["top_k"],
                       seed=gcfg["seed"])
    print(f"\nGeneration (prompt={gcfg['prompt']!r}, temp={gcfg['temperature']}, "
          f"top_k={gcfg['top_k']}):")
    if tok is not None:
        print(f"  {tok.decode(out_ids)}")
    else:
        print(f"  ids: {out_ids}")
    print("  (small model + tiny corpus + few steps -> expect largely incoherent text)")

    rec = {
        "record_type": "evaluation",
        "evaluated_at": date.today().isoformat(),
        "checkpoint": args.checkpoint,
        "batch": args.batch,
        "eval": {"holdout_loss": round(loss, 4), "perplexity": round(ppl, 2)},
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest = REPO / "01_training_input_data" / "manifests" / "training_runs.jsonl"
    with open(manifest, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\nLogged evaluation to {manifest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
