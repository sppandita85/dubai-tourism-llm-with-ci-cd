#!/usr/bin/env python
"""Demonstrate the full model forward pass on a real tokenized batch.

Flow: tokens.bin -> embeddings -> N transformer blocks -> final norm -> logits.
Also computes the next-token cross-entropy, which for random (untrained) weights should
sit near ln(vocab_size) — a sanity check that the untrained model is at chance.

    python 06_transformer_block/scripts/demo.py 2026-07
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent      # transformer_block/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPTModel, cross_entropy  # noqa: E402


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", batch):
        print(f"Batch id must be YYYY-MM (got: {batch!r})", file=sys.stderr)
        return 2

    cfg = yaml.safe_load((ROOT / "config" / "model.yaml").read_text())
    proc = REPO / "01_training_input_data" / "processed" / batch
    if not (proc / "tokens.bin").exists():
        print(f"Missing {proc/'tokens.bin'}. Run tokenization for {batch} first.",
              file=sys.stderr)
        return 1

    meta = json.loads((proc / "meta.json").read_text())
    vocab_size = meta["vocab_size"]
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    ids = np.fromfile(proc / "tokens.bin", dtype=dtype).astype(np.int64)

    ctx = cfg["context_length"]
    batch_size = 4
    # Input/target pairs: target is the input shifted one token to the left.
    x = np.stack([ids[i * ctx:(i + 1) * ctx] for i in range(batch_size)])
    y = np.stack([ids[i * ctx + 1:(i + 1) * ctx + 1] for i in range(batch_size)])

    model = GPTModel(
        vocab_size=vocab_size, emb_dim=cfg["emb_dim"], context_length=ctx,
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        hidden_mult=cfg["hidden_mult"], activation=cfg["activation"],
        dropout=cfg["dropout"], bias=cfg["bias"], pos_type=cfg["pos_type"],
        seed=cfg["seed"],
    )
    logits = model(x)
    loss = cross_entropy(logits, y)

    print(f"Batch {batch}  —  full model forward pass")
    print(f"  config: emb_dim={cfg['emb_dim']}, num_layers={cfg['num_layers']}, "
          f"num_heads={cfg['num_heads']}, vocab_size={vocab_size}")
    print(f"  total parameters : {model.num_parameters:,}")
    print()
    print(f"  input  token ids : {x.shape}   (batch, seq_len)")
    print(f"  output logits    : {logits.shape}   (batch, seq_len, vocab_size)")
    print()
    print(f"  next-token cross-entropy loss : {loss:.4f}")
    print(f"  ln(vocab_size) (random baseline): {math.log(vocab_size):.4f}  "
          f"<- untrained model should be close to this")
    print()
    preds = logits[0, :8].argmax(axis=-1)
    print(f"  argmax next-token predictions, seq 0, positions 0-7 : {preds.tolist()}")
    print(f"  (these are gibberish until the model is trained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
