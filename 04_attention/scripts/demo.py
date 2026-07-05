#!/usr/bin/env python
"""Demonstrate causal multi-head attention on real pipeline data.

Flow: tokens.bin -> input embeddings (embedding phase) -> causal self-attention.

    python 04_attention/scripts/demo.py 2026-07
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent      # attention/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "03_embeddings" / "src"))
from attention import CausalMultiHeadAttention  # noqa: E402
from input_embedding import InputEmbedding       # noqa: E402


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", batch):
        print(f"Batch id must be YYYY-MM (got: {batch!r})", file=sys.stderr)
        return 2

    acfg = yaml.safe_load((ROOT / "config" / "attention.yaml").read_text())
    ecfg = yaml.safe_load((REPO / "03_embeddings" / "config" / "embedding.yaml").read_text())
    proc = REPO / "01_training_input_data" / "processed" / batch
    if not (proc / "tokens.bin").exists():
        print(f"Missing {proc/'tokens.bin'}. Run tokenization for {batch} first.",
              file=sys.stderr)
        return 1

    meta = json.loads((proc / "meta.json").read_text())
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    ids = np.fromfile(proc / "tokens.bin", dtype=dtype).astype(np.int64)

    ctx, emb_dim = ecfg["context_length"], ecfg["emb_dim"]
    x_ids = np.stack([ids[i * ctx:(i + 1) * ctx] for i in range(4)])  # (4, ctx)

    emb = InputEmbedding(meta["vocab_size"], emb_dim, ctx,
                         pos_type=ecfg["pos_type"], dropout=0.0, seed=ecfg["seed"])
    x = emb(x_ids)                                                    # (4, ctx, emb_dim)

    attn = CausalMultiHeadAttention(emb_dim, acfg["num_heads"],
                                    dropout=acfg["dropout"], bias=acfg["bias"],
                                    seed=acfg["seed"])
    y = attn(x)                                                      # (4, ctx, emb_dim)

    print(f"Batch {batch}")
    print(f"  config: emb_dim={emb_dim}, num_heads={acfg['num_heads']}, "
          f"head_dim={emb_dim // acfg['num_heads']}")
    print(f"  learnable parameters : {attn.num_parameters:,}")
    print()
    print(f"  input  (embeddings) shape : {x.shape}")
    print(f"  output (attention)  shape : {y.shape}")
    print()

    # Causal check on the attention weights of the first sequence, head 0.
    W = attn.last_attn[0, 0]                                          # (ctx, ctx)
    upper = W[np.triu_indices_from(W, k=1)]
    print(f"  attention weight matrix (seq {W.shape[0]}x{W.shape[0]}), head 0:")
    print(f"    row 0 attends to positions with weight>0: "
          f"{int((W[0] > 0).sum())}  (expect 1 — only itself)")
    print(f"    row 5 attends to positions with weight>0: "
          f"{int((W[5] > 0).sum())}  (expect 6 — itself + 5 earlier)")
    print(f"    max weight on any FUTURE position         : {float(upper.max()):.2e}  "
          f"(expect 0 — causal mask)")
    print(f"    each row sums to 1?                        : "
          f"{np.allclose(W.sum(axis=1), 1.0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
