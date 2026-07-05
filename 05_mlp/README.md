# Position-Wise Feed-Forward (MLP) Phase

The second sublayer of a transformer block. Where attention mixes information *across*
positions, the MLP transforms **each position independently** through an
expand → nonlinearity → project bottleneck, giving the model capacity to reshape features:

```
hidden = gelu(x @ W1 + b1)     # emb_dim -> hidden (= 4 * emb_dim)
out    = hidden @ W2 + b2      # hidden  -> emb_dim
```

Shape is preserved: `(batch, seq, emb_dim) -> (batch, seq, emb_dim)`.

## Model component, not data prep
Two learned linear layers (weights start random, trained later). NumPy implementation; the
GELU used is the GPT-2 tanh approximation. `relu` is also available via config.

## Layout
```
mlp/
├── config/mlp.yaml       # emb_dim, hidden_mult, activation, dropout, bias, seed
├── src/mlp.py            # FeedForward (NumPy) + gelu/relu
├── scripts/demo.py       # tokens -> embeddings -> attention -> MLP, on a real batch
└── tests/test_mlp.py     # shape / position-wise / GELU / param-count checks
```

## Run
```bash
PY=05_mlp/.venv/bin/python
$PY 05_mlp/scripts/demo.py 2026-07     # demo on real pipeline data
$PY 05_mlp/tests/test_mlp.py           # tests
```

## Notes
- **`hidden_mult=4`** is the GPT convention (hidden = 4×emb_dim).
- The demo runs the MLP *after* attention because that is its position in a transformer
  block — but the MLP itself only requires `(…, emb_dim)` input and is order-independent
  shape-wise. The tests confirm it is genuinely **position-wise**: changing one position
  never affects another.

With defaults (`emb_dim=256`, `hidden_mult=4`, bias on) the MLP has
`2 × 256 × 1024 + 1024 + 256 ≈ 525K` parameters — typically the largest single block in a
small transformer.
