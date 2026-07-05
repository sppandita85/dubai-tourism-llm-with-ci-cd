# Input Embedding Phase (token + positional)

Turns the integer **token IDs** from the tokenization phase into the continuous vectors the
transformer consumes:

```
input_embedding[b, t] = token_embedding[token_id]  +  positional_embedding[t]
```

## How this differs from the tokenization phase
Tokenization is **data prep** — it writes reusable token-ID files to disk per month.
Embeddings are a **model component**: two weight tables that start random and are **learned
during training**. So this folder holds **code** (a reusable `InputEmbedding` class) plus a
**demonstration**, not new monthly data artifacts.

## Why NumPy (not PyTorch)
This machine is an Intel Mac (x86_64) on Python 3.13, and PyTorch no longer ships wheels for
that combination. The embedding is implemented in plain **NumPy** so the lookup-and-add math
is fully visible and runs anywhere. When training moves to a supported environment (Apple
Silicon / Linux / Python ≤3.12), these arrays can be ported to a framework with autograd.

## Positional embedding types (set in [config/embedding.yaml](config/embedding.yaml))
- `learned` (default): a `[context_length, emb_dim]` trainable table — GPT-2 style.
- `sinusoidal`: a fixed sin/cos table from *Attention Is All You Need* (no parameters).
- RoPE is intentionally **not** here — it is applied *inside attention*, not added to inputs,
  so it belongs to the later attention phase.

## Layout
```
embeddings/
├── config/embedding.yaml     # emb_dim, context_length, pos_type, dropout, seed
├── src/input_embedding.py    # the InputEmbedding component (NumPy)
├── scripts/demo.py           # runs it on 01_training_input_data/processed/<batch>/tokens.bin
└── tests/test_embedding.py   # shape / value / positional / guard checks
```
`vocab_size` is not configured here — it is read from the batch's `processed/<batch>/meta.json`
so it always matches the tokenizer.

## Setup (once)
```bash
python3 -m venv 03_embeddings/.venv
03_embeddings/.venv/bin/pip install numpy pyyaml
```

## Run
```bash
PY=03_embeddings/.venv/bin/python
$PY 03_embeddings/scripts/demo.py 2026-07   # demo on the tokenized batch
$PY 03_embeddings/tests/test_embedding.py   # tests
```

## Output
`InputEmbedding.forward(token_ids)` maps `(batch, seq_len)` int IDs to
`(batch, seq_len, emb_dim)` float32 vectors. With the defaults (`emb_dim=256`,
`context_length=256`, `vocab_size=30522`) the learned variant has
`(30522 + 256) × 256 ≈ 7.9M` parameters — the model's first learnable layer.
