"""Export a trained checkpoint (model.npz) to a GGUF file Ollama can run.

Our model is a GPT-2-style decoder (learned positional embeddings, pre-norm blocks,
multi-head attention + bias, GELU MLP + bias, final norm, output head). That maps onto
llama.cpp's `gpt2` architecture, so we write a gpt2 GGUF: hyperparameters, the tokenizer
copied from the nomic-embed-text GGUF (the tokenizer the model was trained with), and the
weight tensors in llama.cpp's expected layout.

Weight layout note: ggml computes `mul_mat(W, x)` as `Wᵀx`, and a numpy array of shape
(rows, cols) becomes ggml ne=[cols, rows]. Our linear weights are (in, out) used as `x@W`,
so each is written transposed → numpy (out, in) → ggml ne=[in, out], which is what
llama.cpp wants. Q/K/V are fused into one `attn_qkv` tensor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter


def _nomic_tokenizer_blob() -> Path:
    man = Path(os.path.expanduser(
        "~/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest"))
    if not man.exists():
        raise FileNotFoundError(f"nomic-embed-text manifest not found: {man}")
    meta = json.loads(man.read_text())
    digest = next(l["digest"] for l in meta["layers"]
                  if l["mediaType"] == "application/vnd.ollama.image.model")
    return Path(os.path.expanduser("~/.ollama/models/blobs")) / digest.replace(":", "-")


def _copy_tokenizer(writer: GGUFWriter) -> None:
    r = GGUFReader(str(_nomic_tokenizer_blob()))
    toks = r.get_field("tokenizer.ggml.tokens")
    tokens = [toks.contents(i) for i in range(len(toks.data))]
    tt = r.get_field("tokenizer.ggml.token_type")
    token_types = [int(tt.parts[tt.data[i]][0]) for i in range(len(tt.data))]
    writer.add_tokenizer_model("bert")          # WordPiece (BERT-style)
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(101)                # [CLS]
    writer.add_eos_token_id(102)                # [SEP]
    writer.add_unk_token_id(100)                # [UNK]
    writer.add_pad_token_id(0)                  # [PAD]


def _T(a: np.ndarray) -> np.ndarray:
    """Transpose an (in, out) weight to (out, in) so ggml ne becomes [in, out]."""
    return np.ascontiguousarray(a.T.astype(np.float32))


def export_checkpoint(checkpoint: str | Path, out_path: str | Path) -> dict:
    """Write a gpt2 GGUF from a checkpoint. Returns the model config dict."""
    data = np.load(str(checkpoint))
    cfg = json.loads(bytes(data["__config__"]).decode("utf-8"))
    P = {k: data[k].astype(np.float32) for k in data.files if k != "__config__"}
    E, L, H = cfg["emb_dim"], cfg["num_layers"], cfg["num_heads"]
    FF = E * cfg["hidden_mult"]

    w = GGUFWriter(str(out_path), "gpt2")
    w.add_name("llm-stepbystep")
    w.add_context_length(cfg["context_length"])
    w.add_embedding_length(E)
    w.add_block_count(L)
    w.add_feed_forward_length(FF)
    w.add_head_count(H)
    w.add_layer_norm_eps(1e-5)
    _copy_tokenizer(w)

    w.add_tensor("token_embd.weight", np.ascontiguousarray(P["emb.tok_emb"]))
    w.add_tensor("position_embd.weight", np.ascontiguousarray(P["emb.pos_emb"]))
    for i in range(L):
        b = f"block{i}"
        w.add_tensor(f"blk.{i}.attn_norm.weight", P[f"{b}.norm1.gamma"])
        w.add_tensor(f"blk.{i}.attn_norm.bias", P[f"{b}.norm1.beta"])
        qkv = np.concatenate(
            [P[f"{b}.attn.W_q"], P[f"{b}.attn.W_k"], P[f"{b}.attn.W_v"]], axis=1)
        qkv_b = np.concatenate(
            [P[f"{b}.attn.b_q"], P[f"{b}.attn.b_k"], P[f"{b}.attn.b_v"]])
        w.add_tensor(f"blk.{i}.attn_qkv.weight", _T(qkv))
        w.add_tensor(f"blk.{i}.attn_qkv.bias", np.ascontiguousarray(qkv_b))
        w.add_tensor(f"blk.{i}.attn_output.weight", _T(P[f"{b}.attn.W_o"]))
        w.add_tensor(f"blk.{i}.attn_output.bias", P[f"{b}.attn.b_o"])
        w.add_tensor(f"blk.{i}.ffn_norm.weight", P[f"{b}.norm2.gamma"])
        w.add_tensor(f"blk.{i}.ffn_norm.bias", P[f"{b}.norm2.beta"])
        w.add_tensor(f"blk.{i}.ffn_up.weight", _T(P[f"{b}.ff.W1"]))
        w.add_tensor(f"blk.{i}.ffn_up.bias", P[f"{b}.ff.b1"])
        w.add_tensor(f"blk.{i}.ffn_down.weight", _T(P[f"{b}.ff.W2"]))
        w.add_tensor(f"blk.{i}.ffn_down.bias", P[f"{b}.ff.b2"])
    w.add_tensor("output_norm.weight", P["final_norm.gamma"])
    w.add_tensor("output_norm.bias", P["final_norm.beta"])
    w.add_tensor("output.weight", _T(P["out_head"]))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return cfg
