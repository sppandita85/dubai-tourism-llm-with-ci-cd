# Training Data

This folder is organized for **monthly, incremental retraining**. Each month you drop
new source documents into a dated batch, run the prep pipeline, and record the batch in a
manifest. Because every batch is dated and tracked, training runs are reproducible and you
always know exactly which data a given model version saw.

## Layout

```
01_training_input_data/
├── raw/            # Untouched source documents, exactly as collected. NEVER edit in place.
│   └── YYYY-MM/    #   One folder per monthly batch, e.g. 2026-07/
├── interim/        # Cleaned / normalized plain text (dedup, stripped markup, fixed encoding).
│   └── YYYY-MM/    #   Output of the cleaning step. Safe to delete & regenerate from raw/.
├── processed/      # Final training-ready artifacts (tokenized shards, .bin/.npy, etc.).
│   └── YYYY-MM/    #   What the trainer actually reads. Regenerable from interim/.
├── holdout/        # A frozen validation/eval set kept OUT of training, to measure each
│                   #   month's model on a stable benchmark. Set once, don't add training data here.
└── manifests/
    ├── dataset_manifest.jsonl   # One line per ingested batch (provenance, hashes, counts).
    └── training_runs.jsonl      # One line per training run (which batches, model version).
```

## Monthly workflow

1. **Ingest** — create `01_training_input_data/raw/YYYY-MM/` and drop this month's new documents in it.
   Keep raw files immutable; if a doc changes, add a new version, don't overwrite.
2. **Clean** — normalize each raw file into `01_training_input_data/interim/YYYY-MM/` (UTF-8 plain text,
   dedup against prior months, remove boilerplate).
3. **Tokenize/pack** — produce training shards into `01_training_input_data/processed/YYYY-MM/`.
4. **Record** — append a row to `manifests/dataset_manifest.jsonl` describing the batch.
5. **Train** — train (from scratch, or continued) on the union of processed batches you
   choose, then append a row to `manifests/training_runs.jsonl`.
6. **Evaluate** — score the new model on `01_training_input_data/holdout/` and log the metric in the run row.

## Conventions

- Batch id = the month folder name, `YYYY-MM` (e.g. `2026-08`).
- Never train on `holdout/`. Never move holdout content into `raw/`.
- `raw/` is the source of truth; `interim/` and `processed/` are always regenerable.
- Record content hashes in the manifest so you can detect accidental duplicate ingestion.

## Full retrain vs. continued training

Because batches are additive and dated, you can support either strategy from the same layout:
- **Full retrain from scratch each month:** train on ALL `processed/*` batches.
- **Continued/incremental training:** train the previous checkpoint on just the newest batch.
The `training_runs.jsonl` `batches` field records exactly which batches each run consumed.
