#!/usr/bin/env python
"""Extract the tokenizer from the Ollama model's GGUF blob into 02_tokenization/vocab/.

The Ollama HTTP server cannot return token IDs, so we read the tokenizer vocabulary
straight out of the model's local GGUF file and rebuild an equivalent HuggingFace
WordPiece tokenizer. Run once per tokenizer (idempotent).

    python 02_tokenization/scripts/extract_vocab.py
"""
from __future__ import annotations

import json
import sys

from gguf import GGUFReader
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from _common import load_config, resolve_model_blob, vocab_dir

WORD_PREFIX = "▁"  # ▁ marks word-initial tokens in llama.cpp's WordPiece dump
TYPE_CONTROL = 3        # tokenizer.ggml.token_type value for special tokens


def _read_tokens(reader: GGUFReader) -> tuple[list[str], list[int]]:
    tok_field = reader.get_field("tokenizer.ggml.tokens")
    type_field = reader.get_field("tokenizer.ggml.token_type")
    n = len(tok_field.data)
    tokens = [tok_field.contents(i) for i in range(n)]
    types = [int(type_field.parts[type_field.data[i]][0]) for i in range(n)]
    return tokens, types


def _to_hf_vocab_token(tok: str, ttype: int) -> str:
    """Convert a GGUF WordPiece token to the HuggingFace bert vocab.txt form.

    ▁word  -> word        (word-initial piece, bare in HF)
    [CTRL] -> [CTRL]       (special/control tokens kept verbatim)
    piece  -> ##piece      (continuation piece, ## prefixed in HF)
    """
    if tok.startswith(WORD_PREFIX):
        return tok[len(WORD_PREFIX):]
    if ttype == TYPE_CONTROL or (tok.startswith("[") and tok.endswith("]")):
        return tok
    return "##" + tok


def _special_id(reader: GGUFReader, key: str, default: int | None = None):
    f = reader.get_field(key)
    return f.contents() if f is not None else default


def main() -> int:
    cfg = load_config()
    blob, digest = resolve_model_blob(cfg)
    print(f"Reading tokenizer from GGUF blob:\n  {blob}")
    reader = GGUFReader(str(blob))

    model_type = reader.get_field("tokenizer.ggml.model").contents()
    if model_type != "bert":
        print(f"WARNING: expected a 'bert' WordPiece tokenizer, got '{model_type}'. "
              "This extractor targets WordPiece; results may be wrong.", file=sys.stderr)

    tokens, types = _read_tokens(reader)
    vocab = {}
    collisions = 0
    for i, (tok, ttype) in enumerate(zip(tokens, types)):
        hf = _to_hf_vocab_token(tok, ttype)
        if hf in vocab:  # should not happen for a well-formed vocab
            collisions += 1
        vocab[hf] = i
    if collisions:
        print(f"WARNING: {collisions} token string collisions during reconstruction",
              file=sys.stderr)

    specials = {
        "unk": _special_id(reader, "tokenizer.ggml.unknown_token_id", 100),
        "cls": _special_id(reader, "tokenizer.ggml.cls_token_id", 101),
        # note: GGUF key is spelled 'seperator' in this model
        "sep": _special_id(reader, "tokenizer.ggml.seperator_token_id",
                           _special_id(reader, "tokenizer.ggml.sep_token_id", 102)),
        "pad": _special_id(reader, "tokenizer.ggml.padding_token_id", 0),
        "mask": _special_id(reader, "tokenizer.ggml.mask_token_id", 103),
    }
    unk_tok = tokens[specials["unk"]] if specials["unk"] is not None else "[UNK]"
    cls_tok = cfg["special_tokens"]["cls"]
    sep_tok = cfg["special_tokens"]["sep"]

    # Build an equivalent HuggingFace WordPiece tokenizer (bert-base-uncased behaviour).
    tokenizer = Tokenizer(WordPiece(vocab, unk_token=unk_tok, max_input_chars_per_word=100))
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{cls_tok} $A {sep_tok}",
        pair=f"{cls_tok} $A {sep_tok} $B:1 {sep_tok}:1",
        special_tokens=[(cls_tok, specials["cls"]), (sep_tok, specials["sep"])],
    )
    # Flag the control tokens as special so decode(skip_special_tokens=True) drops them.
    # These already exist in the vocab, so their IDs are preserved.
    tokenizer.add_special_tokens(list(cfg["special_tokens"].values()))

    out = vocab_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.txt").write_text(
        "\n".join(tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])) + "\n"
    )
    tokenizer.save(str(out / "tokenizer.json"))
    meta = {
        "source_model": cfg["source_model"],
        "source_blob_digest": digest,
        "tokenizer_model_type": model_type,
        "vocab_size": len(vocab),
        "special_tokens": specials,
        "word_prefix_marker": "U+2581",
    }
    (out / "vocab_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"vocab_size = {len(vocab)}")
    print(f"special tokens = {specials}")
    print(f"Wrote: {out}/vocab.txt, tokenizer.json, vocab_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
