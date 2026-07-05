# Evaluation Phase

Measures how good a trained checkpoint is, two ways:
1. **Perplexity** on held-out tokens — the standard quantitative metric (`exp(cross-entropy)`;
   lower is better, a random model scores ≈ vocab_size).
2. **Text generation** — a qualitative sample, decoded back to words with the tokenizer.

This is what tells you whether *this* month's model actually improved over last month's — so
run it on the same held-out data each month for a comparable number.

## Layout
```
evaluation/
├── config/eval.yaml         # perplexity window count + generation settings
├── src/
│   ├── evaluate.py          # perplexity / loss over held-out tokens
│   └── generate.py          # autoregressive sampling (temperature, top-k)
├── scripts/evaluate.py      # CLI: load checkpoint -> perplexity + sample -> log
└── tests/test_evaluation.py # perplexity==exp(loss), generation validity, greedy determinism
```

## Run
```bash
PY=08_evaluation/.venv/bin/python
$PY 08_evaluation/scripts/evaluate.py --checkpoint checkpoints/v0.1.0/model.npz --batch 2026-07
$PY 08_evaluation/tests/test_evaluation.py
```
It evaluates on the held-out tail of the batch (or the dedicated `holdout/` set once that is
tokenized), decodes a generated sample via `02_tokenization/vocab/…/tokenizer.json`, and appends
an evaluation record to `01_training_input_data/manifests/training_runs.jsonl`.

## Reading the numbers
- **Perplexity ≈ vocab_size (~30k):** the model is at chance (untrained).
- **Perplexity falling month over month:** the model is learning.
- Expect **incoherent generated text** here: a 2-layer model trained for a few hundred CPU
  steps on one document cannot produce fluent prose. The point is that the *machinery* works
  end to end; fluency needs a bigger model, more data, and more steps (i.e. GPU/PyTorch).
