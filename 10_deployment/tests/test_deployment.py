#!/usr/bin/env python
"""Deployment-phase tests: a checkpoint exports to a valid gpt2 GGUF.

Verifies the GGUF is well-formed and has the architecture/metadata/tensors Ollama needs.
Does not require Ollama (the real `ollama create` is exercised by the end-to-end run).

    python 10_deployment/tests/test_deployment.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))
from gguf import GGUFReader             # noqa: E402
from model import GPTModel              # noqa: E402
from export_gguf import export_checkpoint  # noqa: E402


def main() -> int:
    failures = []
    V, E, C, H, L = 30522, 32, 16, 2, 2   # real vocab (tokenizer copy needs nomic blob)
    model = GPTModel(vocab_size=V, emb_dim=E, context_length=C, num_heads=H,
                     num_layers=L, hidden_mult=4, dropout=0.0, seed=0)

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "model.npz"
        gguf = Path(td) / "model.gguf"
        model.save(str(ckpt))
        try:
            export_checkpoint(ckpt, gguf)
        except FileNotFoundError as e:
            print(f"SKIP: nomic-embed-text GGUF not available ({e})")
            return 0

        r = GGUFReader(str(gguf))
        kv = {f.name: f for f in r.fields.values()}
        names = {t.name for t in r.tensors}

        if kv["general.architecture"].contents() != "gpt2":
            failures.append("architecture is not gpt2")
        if kv["gpt2.block_count"].contents() != L:
            failures.append(f"block_count != {L}")
        if kv["gpt2.embedding_length"].contents() != E:
            failures.append(f"embedding_length != {E}")
        if kv["gpt2.attention.head_count"].contents() != H:
            failures.append(f"head_count != {H}")
        if kv["tokenizer.ggml.model"].contents() != "bert":
            failures.append("tokenizer model is not bert")

        required = {"token_embd.weight", "position_embd.weight",
                    "output_norm.weight", "output.weight"}
        for i in range(L):
            required |= {f"blk.{i}.attn_qkv.weight", f"blk.{i}.attn_output.weight",
                         f"blk.{i}.ffn_up.weight", f"blk.{i}.ffn_down.weight",
                         f"blk.{i}.attn_norm.weight", f"blk.{i}.ffn_norm.weight"}
        missing = required - names
        if missing:
            failures.append(f"missing tensors: {sorted(missing)}")

        # Fused QKV must be 3x the embedding on its output axis.
        qkv = next(t for t in r.tensors if t.name == "blk.0.attn_qkv.weight")
        if tuple(qkv.shape) != (E, 3 * E):    # gguf reports ne = [in, out] = [E, 3E]
            failures.append(f"attn_qkv shape {tuple(qkv.shape)} != {(E, 3*E)}")

    if failures:
        print("DEPLOYMENT TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All deployment tests passed.")
    print("  exported a valid gpt2 GGUF with bert tokenizer and fused QKV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
