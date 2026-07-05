#!/usr/bin/env python
"""Tokenize a cleaned batch into a flat token-ID stream for training.

Reads 01_training_input_data/interim/<batch>/*.txt, encodes each document with the
tokenizer extracted from the Ollama model (see extract_vocab.py), concatenates the
token IDs into one array, and writes:

    01_training_input_data/processed/<batch>/tokens.bin   (raw uint16/uint32 IDs)
    01_training_input_data/processed/<batch>/meta.json    (counts, dtype, offsets)
    02_tokenization/logs/<batch>.json                     (run report)

It also appends a tokenization record to the dataset manifest.

    python 02_tokenization/scripts/tokenize_batch.py 2026-07
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timezone

import numpy as np
from tokenizers import Tokenizer

from _common import DATA_ROOT, TOK_ROOT, batch_dir, load_config, vocab_dir

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", batch):
        print(f"Batch id must be YYYY-MM (got: {batch!r})", file=sys.stderr)
        return 2

    cfg = load_config()
    vdir = vocab_dir(cfg)
    tok_json = vdir / "tokenizer.json"
    if not tok_json.exists():
        print(f"Tokenizer not found: {tok_json}\n"
              "Run: python 02_tokenization/scripts/extract_vocab.py first.", file=sys.stderr)
        return 1
    tokenizer = Tokenizer.from_file(str(tok_json))
    vocab_size = tokenizer.get_vocab_size()

    dtype_name = cfg.get("output_dtype", "uint16")
    if dtype_name == "uint16" and vocab_size >= 2 ** 16:
        print(f"vocab_size {vocab_size} too large for uint16; using uint32", file=sys.stderr)
        dtype_name = "uint32"
    dtype = DTYPES[dtype_name]

    src = batch_dir("interim", batch)
    if not src.is_dir():
        print(f"No interim batch folder: {src}\n"
              "Run: python 02_tokenization/scripts/clean_batch.py <batch> first.",
              file=sys.stderr)
        return 1
    files = sorted(src.glob("*.txt"))
    if not files:
        print(f"No .txt files in {src}", file=sys.stderr)
        return 1

    all_ids: list[int] = []
    offsets = []          # per-document [start, end) into the token stream
    unk_id = tokenizer.token_to_id(cfg["special_tokens"]["unk"])
    unk_count = 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids  # [CLS] ... [SEP] via post-processor
        start = len(all_ids)
        all_ids.extend(ids)
        offsets.append({"file": p.name, "start": start, "end": len(all_ids),
                        "n_tokens": len(ids)})
        unk_count += sum(1 for t in ids if t == unk_id)

    arr = np.asarray(all_ids, dtype=dtype)
    dst = batch_dir("processed", batch)
    dst.mkdir(parents=True, exist_ok=True)
    bin_path = dst / "tokens.bin"
    arr.tofile(bin_path)

    n_tokens = int(arr.size)
    unk_rate = (unk_count / n_tokens) if n_tokens else 0.0
    meta = {
        "batch": batch,
        "n_tokens": n_tokens,
        "n_documents": len(files),
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "tokenizer_source": cfg["source_model"],
        "tokenizer_vocab_dir": str(vdir.relative_to(TOK_ROOT.parent)),
        "unk_token_id": unk_id,
        "unk_count": unk_count,
        "unk_rate": round(unk_rate, 6),
        "documents": offsets,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    log_dir = TOK_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{batch}.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Append a tokenization record to the dataset manifest (one line).
    manifest = DATA_ROOT / "manifests" / "dataset_manifest.jsonl"
    record = {
        "record_type": "tokenization",
        "batch": batch,
        "tokenized_at": date.today().isoformat(),
        "tokenizer_source": cfg["source_model"],
        "vocab_size": vocab_size,
        "n_tokens": n_tokens,
        "n_documents": len(files),
        "output": str(bin_path.relative_to(TOK_ROOT.parent)),
        "unk_rate": round(unk_rate, 6),
    }
    with open(manifest, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"Tokenized {len(files)} doc(s) in batch {batch}")
    print(f"  vocab_size = {vocab_size}, dtype = {dtype_name}")
    print(f"  total tokens = {n_tokens:,}  (unknown-token rate = {unk_rate:.4%})")
    print(f"  wrote {bin_path} ({bin_path.stat().st_size:,} bytes) + meta.json")
    print(f"  first 20 token IDs: {arr[:20].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
