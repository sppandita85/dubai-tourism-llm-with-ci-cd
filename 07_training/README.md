# Training Phase

Learns the model weights: **forward → loss → backward → optimizer step**, repeated. This is
the phase that turns the random-init model (loss ≈ `ln(vocab)`) into one that predicts the
next token. It is the heart of the monthly retraining loop.

## What's here
```
training/
├── config/train.yaml     # model size + training hyperparameters
├── src/
│   ├── dataloader.py      # tokens.bin -> (input, target) batches; train/val split
│   ├── optimizer.py       # AdamW (decoupled weight decay), NumPy
│   └── trainer.py         # the train loop + validation-loss helper
├── scripts/train.py       # CLI: train from scratch on a batch, checkpoint, log the run
└── tests/test_training.py # AdamW + OVERFIT proof + checkpoint round-trip
```

## The backward pass
The components from the earlier phases (`embeddings/`, `attention/`, `mlp/`,
`transformer_block/`) each gained a `backward()` method and expose `parameters()` /
`gradients()`. `GPTModel.loss_and_backward()` runs the full chain:
cross-entropy → output head → final norm → blocks (reversed) → embeddings. It was verified
two ways:
1. **Numerical gradient check** (analytic vs finite-difference) — matches to ~1e-6 in float64.
2. **Overfit test** — a small model memorizes one fixed batch, loss `ln(V)` → ~0.01. A wrong
   gradient cannot do this. (Run `tests/test_training.py`.)

## Run
```bash
PY=07_training/.venv/bin/python
$PY 07_training/tests/test_training.py            # verify backward + optimizer
$PY 07_training/scripts/train.py 2026-07          # train from scratch on a batch
$PY 07_training/scripts/train.py 2026-07 --steps 500 --version v0.2.0
```
Outputs a checkpoint at `checkpoints/<version>/model.npz` and appends a row to
`01_training_input_data/manifests/training_runs.jsonl`.

## A necessary caveat on scale
This is **pure NumPy on CPU** (forced by the Intel-Mac / Python 3.13 environment where
PyTorch has no wheels). It is correct and great for learning, but slow: the config ships a
small model (2 layers, emb_dim 128) so a run finishes in minutes on the tiny corpus. For
real-scale training, move to a machine with PyTorch/GPU and port these components (the math
is identical); the tokenized `tokens.bin` files carry over unchanged.
