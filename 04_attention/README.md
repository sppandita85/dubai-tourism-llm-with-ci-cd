# Causal Multi-Head Self-Attention Phase

The layer where the model **mixes information across positions**. Each position emits a
query, key, and value; attention scores decide how much each position pulls from the
others — restricted to **itself and earlier positions** (the causal mask), which is what
makes the model autoregressive and able to generate left-to-right.

```
x (batch, seq, emb) ─▶ Q,K,V projections ─▶ scaled dot-product ─▶ causal mask
                     ─▶ softmax ─▶ weighted sum of V ─▶ merge heads ─▶ output proj ─▶ y
```

## Model component, not data prep
Like the embedding phase, this is a **learned layer** (the Q/K/V/output projection weights
start random and are trained). The folder holds the `CausalMultiHeadAttention` class plus a
demo — no per-month data artifacts. Implemented in **NumPy** (see the embedding phase README
for why), so masking and softmax are fully explicit.

## Layout
```
attention/
├── config/attention.yaml    # emb_dim, num_heads, dropout, bias, seed
├── src/attention.py         # CausalMultiHeadAttention (NumPy)
├── scripts/demo.py          # tokens -> embeddings -> attention, on a real batch
└── tests/test_attention.py  # shape / causality / mask / param-count checks
```
`emb_dim` and `num_heads` must satisfy `emb_dim % num_heads == 0` (head_dim = emb_dim / num_heads).

## Run
```bash
PY=04_attention/.venv/bin/python
$PY 04_attention/scripts/demo.py 2026-07     # demo on real pipeline data
$PY 04_attention/tests/test_attention.py     # tests
```

## Key property it guarantees
The causal mask sets attention weights on any **future** position to zero before softmax, so
each output position depends only on the current and earlier tokens. The tests verify this
end-to-end: perturbing a later token leaves all earlier outputs unchanged.

With defaults (`emb_dim=256`, `num_heads=8`, bias on) the layer has `4 × 256² + 4 × 256 ≈
263K` parameters.
