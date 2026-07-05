#!/usr/bin/env python
"""Demonstrate the position-wise MLP on real pipeline data.

Flow: tokens.bin -> input embeddings -> causal self-attention -> MLP.
(The MLP normally follows attention inside a transformer block, so the demo runs the
full sublayer sequence to show real data flowing through both new phases.)

    python 05_mlp/scripts/demo.py 2026-07
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent      # mlp/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "03_embeddings" / "src"))
sys.path.insert(0, str(REPO / "04_attention" / "src"))
from mlp import FeedForward                          # noqa: E402
from input_embedding import InputEmbedding           # noqa: E402
from attention import CausalMultiHeadAttention       # noqa: E402


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", batch):
        print(f"Batch id must be YYYY-MM (got: {batch!r})", file=sys.stderr)
        return 2

    mcfg = yaml.safe_load((ROOT / "config" / "mlp.yaml").read_text())
    ecfg = yaml.safe_load((REPO / "03_embeddings" / "config" / "embedding.yaml").read_text())
    acfg = yaml.safe_load((REPO / "04_attention" / "config" / "attention.yaml").read_text())
    proc = REPO / "01_training_input_data" / "processed" / batch
    if not (proc / "tokens.bin").exists():
        print(f"Missing {proc/'tokens.bin'}. Run tokenization for {batch} first.",
              file=sys.stderr)
        return 1

    meta = json.loads((proc / "meta.json").read_text())
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    ids = np.fromfile(proc / "tokens.bin", dtype=dtype).astype(np.int64)

    ctx, emb_dim = ecfg["context_length"], ecfg["emb_dim"]
    x_ids = np.stack([ids[i * ctx:(i + 1) * ctx] for i in range(4)])

    emb = InputEmbedding(meta["vocab_size"], emb_dim, ctx,
                         pos_type=ecfg["pos_type"], dropout=0.0, seed=ecfg["seed"])
    attn = CausalMultiHeadAttention(emb_dim, acfg["num_heads"],
                                    dropout=0.0, bias=acfg["bias"], seed=acfg["seed"])
    ff = FeedForward(emb_dim, hidden_mult=mcfg["hidden_mult"], activation=mcfg["activation"],
                     dropout=mcfg["dropout"], bias=mcfg["bias"], seed=mcfg["seed"])

    x = emb(x_ids)
    a = attn(x)
    y = ff(a)

    print(f"Batch {batch}")
    print(f"  config: emb_dim={emb_dim}, hidden={ff.hidden} "
          f"(x{mcfg['hidden_mult']}), activation={mcfg['activation']}")
    print(f"  learnable parameters (MLP) : {ff.num_parameters:,}")
    print()
    print(f"  embeddings shape : {x.shape}")
    print(f"  after attention  : {a.shape}")
    print(f"  after MLP        : {y.shape}   (position-wise: shape preserved)")
    print()
    print(f"  MLP input  stats : mean={a.mean():+.4f}, std={a.std():.4f}")
    print(f"  MLP output stats : mean={y.mean():+.4f}, std={y.std():.4f}")
    print(f"  hidden expansion : {emb_dim} -> {ff.hidden} -> {emb_dim} per position")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
