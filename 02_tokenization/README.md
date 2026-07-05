# Tokenization Phase

Turns cleaned text into the integer **token-ID** streams the trainer consumes, using the
tokenizer of a locally-hosted **Ollama** model.

## Why it reads the GGUF instead of calling Ollama
The Ollama HTTP server **cannot return token IDs** — `/api/tokenize` doesn't exist (404),
and `/api/embed` returns an embedding vector plus a token *count* only. So the tokenizer is
read **offline** from the model's GGUF blob under `~/.ollama/models/blobs/…`, and an
equivalent HuggingFace WordPiece tokenizer is rebuilt from it. This yields real token IDs
that match the Ollama model's vocabulary exactly, with no network call.

Default source model: **nomic-embed-text** — a BERT WordPiece tokenizer, 30522-token vocab.
Change it in [config/tokenizer.yaml](config/tokenizer.yaml).

## Layout
```
tokenization/
├── config/tokenizer.yaml   # source model, vocab dir, special tokens, output dtype
├── vocab/<model>/          # extracted tokenizer artifact (vocab.txt, tokenizer.json, meta)
├── scripts/
│   ├── extract_vocab.py    # GGUF -> vocab/            (run once per tokenizer)
│   ├── clean_batch.py      # raw/<batch> -> interim/   (strip markdown)
│   └── tokenize_batch.py   # interim/<batch> -> processed/tokens.bin + meta.json
├── logs/<batch>.json       # per-batch tokenization report
└── tests/test_roundtrip.py # tokenizer sanity checks
```
Inputs come from `01_training_input_data/interim/`, outputs go to
`01_training_input_data/processed/`. The tokenizer is stable across months, so nothing here
is dated — only the data is.

## Setup (once)
```bash
python3 -m venv 02_tokenization/.venv
02_tokenization/.venv/bin/pip install gguf tokenizers numpy pyyaml
```

## Run (per month)
```bash
PY=02_tokenization/.venv/bin/python
$PY 02_tokenization/scripts/extract_vocab.py           # once per tokenizer
$PY 02_tokenization/scripts/clean_batch.py   2026-07   # raw md -> interim txt
$PY 02_tokenization/scripts/tokenize_batch.py 2026-07  # interim txt -> processed tokens.bin
$PY 02_tokenization/tests/test_roundtrip.py            # sanity check
```

## Output format
`processed/<batch>/tokens.bin` is a flat little-endian array of `uint16` token IDs (valid
while vocab_size < 65536), each document wrapped as `[CLS] … [SEP]` and concatenated.
`processed/<batch>/meta.json` records `n_tokens`, `vocab_size`, `dtype`, the tokenizer
source, per-document offsets, and the unknown-token rate. Load it for training with:

```python
import numpy as np
ids = np.fromfile("01_training_input_data/processed/2026-07/tokens.bin", dtype=np.uint16)
```
