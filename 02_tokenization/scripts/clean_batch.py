#!/usr/bin/env python
"""Clean raw source docs for a batch into interim/ plain text.

Reads 01_training_input_data/raw/<batch>/*.md (and *.txt), strips Markdown syntax and
YAML frontmatter, normalizes whitespace to UTF-8 plain text, and writes
01_training_input_data/interim/<batch>/<name>.txt.

    python 02_tokenization/scripts/clean_batch.py 2026-07
    python 02_tokenization/scripts/clean_batch.py          # defaults to current month
"""
from __future__ import annotations

import re
import sys
from datetime import date

from _common import batch_dir

RAW_EXTS = (".md", ".markdown", ".txt")


def strip_markdown(text: str) -> str:
    # YAML frontmatter at the very top.
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # Fenced code blocks -> keep inner text, drop the fences.
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    # Images ![alt](url) -> drop entirely; links [text](url) -> keep text.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Headings, blockquotes, list markers at line start.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}([-*+]|\d+\.)\s+", "", text)
    # Emphasis / inline code markers.
    text = re.sub(r"[*_`~]+", "", text)
    # Collapse 3+ blank lines, trim trailing spaces.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", batch):
        print(f"Batch id must be YYYY-MM (got: {batch!r})", file=sys.stderr)
        return 2

    src = batch_dir("raw", batch)
    if not src.is_dir():
        print(f"No raw batch folder: {src}", file=sys.stderr)
        return 1
    dst = batch_dir("interim", batch)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in RAW_EXTS)
    if not files:
        print(f"No source files ({', '.join(RAW_EXTS)}) in {src}", file=sys.stderr)
        return 1

    for p in files:
        cleaned = strip_markdown(p.read_text(encoding="utf-8"))
        out = dst / (p.stem + ".txt")
        out.write_text(cleaned, encoding="utf-8")
        print(f"cleaned {p.name} -> {out.relative_to(dst.parent.parent.parent)} "
              f"({len(cleaned)} chars)")
    print(f"Done: {len(files)} file(s) -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
