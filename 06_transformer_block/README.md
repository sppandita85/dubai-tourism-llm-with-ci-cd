# Transformer Block Phase (assembled model)

The glue that turns the standalone `attention/` and `mlp/` components into a working
model. It adds the two pieces those phases deliberately left out — **LayerNorm** and
**residual connections** — assembles them into a repeatable **transformer block**, stacks N
of them, and finishes with a final norm + output head to produce next-token logits.

## The block (pre-norm, GPT-2 style)
```
x = x + attention(norm1(x))     # attention sublayer + residual
x = x + mlp(norm2(x))           # feed-forward sublayer + residual
```
- **norm** conditions the copy fed to each sublayer (mean 0 / unit variance per token).
- **residual** (`x + …`) adds the raw input back, so refinements accumulate and deep
  stacks stay trainable.

## The full model
```
token ids ─▶ embeddings ─▶ [ block ] × N ─▶ final norm ─▶ output head ─▶ logits
                                                            (emb_dim → vocab_size)
```
This is the **complete forward pass**: token IDs in, a probability-shaped score over the
whole 30,522-token vocabulary out for every position. Weights are random until the training
phase learns them.

## Layout
```
transformer_block/
├── config/model.yaml          # emb_dim, num_heads, num_layers, hidden_mult, ...
├── src/
│   ├── layer_norm.py          # LayerNorm
│   ├── transformer_block.py   # TransformerBlock (reuses attention/ + mlp/ components)
│   └── model.py               # GPTModel (embeddings + N blocks + head) + cross_entropy
├── scripts/demo.py            # full forward pass on a real batch, with loss sanity check
└── tests/test_model.py        # LayerNorm / causality / param-count / loss checks
```
It imports the components from the sibling phases (`embeddings/`, `attention/`, `mlp/`) via
their `src/` folders — nothing is reimplemented.

## Run
```bash
PY=06_transformer_block/.venv/bin/python
$PY 06_transformer_block/scripts/demo.py 2026-07   # forward pass on tokenized batch
$PY 06_transformer_block/tests/test_model.py       # tests
```

## What "working" looks like here
For an **untrained** model, the next-token cross-entropy loss should sit near
`ln(vocab_size) ≈ 10.33` — i.e. the model is guessing at chance. That is the correct,
expected result; lowering that loss is the job of the **training** phase (next). The demo
prints this comparison, and the tests assert causality holds end-to-end (a later token can
never change an earlier position's logits).
