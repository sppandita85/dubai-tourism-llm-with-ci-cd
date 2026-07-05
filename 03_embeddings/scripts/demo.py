#!/usr/bin/env python
"""Demonstrate the input-embedding phase on a real tokenized batch.

Loads the token IDs produced by the tokenization phase
(01_training_input_data/processed/<batch>/tokens.bin), forms a small batch of input
sequences with a sliding window, and runs them through the InputEmbedding layer to
produce input embeddings of shape (batch, context_length, emb_dim).

    python 03_embeddings/scripts/demo.py 2026-07
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent          # embeddings/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from input_embedding import InputEmbedding  # noqa: E402


def make_batches(ids: np.ndarray, context_length: int, batch_size: int, stride: int):
    """Sliding-window (input, target) pairs; target is input shifted by one token."""
    inputs, targets = [], []
    for start in range(0, len(ids) - context_length, stride):
        inputs.append(ids[start:start + context_length])
        targets.append(ids[start + 1:start + context_length + 1])
        if len(inputs) == batch_size:
            break
    return np.stack(inputs), np.stack(targets)


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", batch):
        print(f"Batch id must be YYYY-MM (got: {batch!r})", file=sys.stderr)
        return 2

    cfg = yaml.safe_load((ROOT / "config" / "embedding.yaml").read_text())
    proc = REPO / "01_training_input_data" / "processed" / batch
    meta_path, bin_path = proc / "meta.json", proc / "tokens.bin"
    if not bin_path.exists():
        print(f"Missing {bin_path}. Run the tokenization phase for {batch} first.",
              file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text())
    vocab_size = meta["vocab_size"]
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    ids = np.fromfile(bin_path, dtype=dtype).astype(np.int64)

    ctx = cfg["context_length"]
    emb_dim = cfg["emb_dim"]
    batch_size = 4
    x, y = make_batches(ids, ctx, batch_size, stride=ctx)

    emb = InputEmbedding(
        vocab_size=vocab_size, emb_dim=emb_dim, context_length=ctx,
        pos_type=cfg["pos_type"], dropout=cfg["dropout"], seed=cfg["seed"],
    )
    out = emb(x, training=False)

    print(f"Batch {batch}")
    print(f"  tokens available         : {len(ids):,}")
    print(f"  vocab_size (from tokenizer): {vocab_size}")
    print(f"  config: emb_dim={emb_dim}, context_length={ctx}, pos_type={cfg['pos_type']}")
    print(f"  learnable parameters      : {emb.num_parameters:,}")
    print()
    print(f"  input  token-id batch shape : {x.shape}   (batch, seq_len)")
    print(f"  target token-id batch shape : {y.shape}")
    print(f"  --> input embedding shape   : {out.shape}   (batch, seq_len, emb_dim)")
    print()
    print(f"  first input sequence, first 12 ids : {x[0, :12].tolist()}")
    print(f"  embedding vector for that first token (first 6 dims):")
    print(f"    {np.round(out[0, 0, :6], 4).tolist()}")

    # Show the positional effect: same token id at two positions -> different vectors.
    tid = int(x[0, 0])
    twopos = emb(np.array([[tid, tid]]))
    diff = float(np.abs(twopos[0, 0] - twopos[0, 1]).mean())
    print()
    print(f"  same token id ({tid}) at positions 0 and 1 differ by mean|Δ| = {diff:.5f}")
    print(f"  (proves positional information is being added, not just token identity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
