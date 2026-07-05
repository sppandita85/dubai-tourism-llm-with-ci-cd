#!/usr/bin/env python
"""Sanity checks for the extracted tokenizer.

WordPiece is lossy (lowercasing, accent stripping, unknown pieces), so we don't assert
byte-exact round-trips. Instead we check that decode(encode(x)) recovers the words of a
normalized sentence, that special IDs are correct, and that a known sentence produces a
stable, non-degenerate token sequence.

    python 02_tokenization/tests/test_roundtrip.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tokenizers import Tokenizer  # noqa: E402

from _common import load_config, vocab_dir  # noqa: E402


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def main() -> int:
    cfg = load_config()
    tok_path = vocab_dir(cfg) / "tokenizer.json"
    if not tok_path.exists():
        print(f"FAIL: {tok_path} missing — run extract_vocab.py first", file=sys.stderr)
        return 1
    tk = Tokenizer.from_file(str(tok_path))
    failures = []

    # 1. Special tokens map to the expected bert-base-uncased IDs.
    expected = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}
    for t, i in expected.items():
        got = tk.token_to_id(t)
        if got != i:
            failures.append(f"special {t}: expected id {i}, got {got}")

    # 2. Vocab size is the full bert vocab.
    if tk.get_vocab_size() != 30522:
        failures.append(f"vocab_size expected 30522, got {tk.get_vocab_size()}")

    # 3. Word-level round-trip on a clean sentence.
    text = "Dubai is the largest city in the United Arab Emirates."
    enc = tk.encode(text)
    dec = tk.decode(enc.ids)  # skips special tokens by default
    if words(dec) != words(text):
        failures.append(f"round-trip words differ:\n  in : {words(text)}\n  out: {words(dec)}")

    # 4. Encoding wraps the doc in [CLS] ... [SEP] and stays non-degenerate.
    if not (enc.ids[0] == 101 and enc.ids[-1] == 102):
        failures.append(f"expected [CLS]..[SEP] wrapping, got {enc.ids[:1]}..{enc.ids[-1:]}")
    unk = tk.token_to_id("[UNK]")
    if any(t == unk for t in enc.ids):
        failures.append("unexpected [UNK] in a plain English sentence")

    if failures:
        print("TOKENIZER TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All tokenizer tests passed.")
    print(f"  sample: {text!r}")
    print(f"  ids   : {enc.ids}")
    print(f"  decode: {tk.decode(enc.ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
