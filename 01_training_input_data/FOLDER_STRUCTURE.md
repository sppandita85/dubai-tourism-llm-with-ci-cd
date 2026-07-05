# Training Input Data — Folder Structure Guide

This document explains how the `01_training_input_data/` folder is organized, why each
sub-folder exists, and walks through a complete worked example of adding one month of
data and training on it.

The whole design exists to support **one goal: retraining the LLM every month, safely and
reproducibly.** Every new month of data lives in its own dated batch, raw sources are kept
immutable, and every ingestion and training run is recorded so you can always answer the
question *"exactly which data did model version X learn from?"*

---

## 1. Top-level map

```
01_training_input_data/
├── raw/            # Immutable source documents, exactly as collected (per month)
├── interim/        # Cleaned / normalized plain text (per month, regenerable)
├── processed/      # Final training-ready artifacts — tokenized shards (per month, regenerable)
├── holdout/        # Frozen evaluation set, never used for training
├── manifests/      # Ledgers: what was ingested, and what each training run consumed
│   ├── dataset_manifest.jsonl
│   └── training_runs.jsonl
├── new_month.sh    # Helper script that scaffolds a new monthly batch
└── README.md       # Short quick-reference version of this guide
```

A **batch** is one month of data. Batches are named `YYYY-MM` (e.g. `2026-07`,
`2026-08`) and the *same* batch id is reused as the sub-folder name inside `raw/`,
`interim/`, and `processed/`. This lets you trace a single month across all three stages.

---

## 2. The data flow (mental model)

Data moves left to right through three stages. Each stage is a separate folder so that
the expensive, hard-to-recreate stuff (`raw/`) is protected, and the cheap, script-generated
stuff (`interim/`, `processed/`) can be deleted and rebuilt at any time.

```
   collect            clean                tokenize / pack
  ─────────▶  raw/  ─────────▶  interim/  ─────────────────▶  processed/  ──▶  TRAINER
 (source of                (regenerable)                  (regenerable)
   truth)
```

- If you change how you clean text, you delete `interim/` + `processed/` and rebuild from `raw/`.
- If you lose `raw/`, the data is gone for good — so `raw/` is the one folder you back up.

---

## 3. Sub-folder reference

### 3.1 `raw/` — the source of truth
- **Contains:** the original documents exactly as you collected them, one sub-folder per month.
- **Rule:** never edit a file here in place. If a document changes, add a new file
  (e.g. `dubai_tourism_v2.md`) — don't overwrite. This is what makes runs reproducible.
- **Back this up.** Everything else can be regenerated; this cannot.
- **Example path:** `raw/2026-07/dubai_tourism.md`

### 3.2 `interim/` — cleaned text
- **Contains:** the raw documents after normalization — converted to UTF-8 plain text,
  markup/boilerplate stripped, whitespace fixed, and de-duplicated against earlier months.
- **Purpose:** a human-readable checkpoint between "messy source" and "tokenized binary."
  You can open these files and confirm the cleaning did the right thing.
- **Regenerable:** produced from `raw/` by your cleaning script; safe to delete.
- **Example path:** `interim/2026-07/dubai_tourism.txt`

### 3.3 `processed/` — training-ready shards
- **Contains:** the final artifacts the trainer actually reads — tokenized sequences packed
  into shards (`.bin`, `.npy`, `.idx`, etc.).
- **Purpose:** fast, fixed-format input for the training loop. Not human-readable.
- **Regenerable:** produced from `interim/` by your tokenizer; safe to delete.
- **Example path:** `processed/2026-07/train_000.bin`

### 3.4 `holdout/` — frozen evaluation set
- **Contains:** a fixed set of documents (or tokenized eval shards) set aside **once** and
  **never trained on.**
- **Purpose:** a stable benchmark. Because it never changes and the model never trains on
  it, comparing this month's model to last month's on the holdout tells you whether the
  model genuinely improved (rather than just memorizing new training data).
- **Rule:** never move holdout content into `raw/`, and don't keep adding to it — a moving
  benchmark can't be compared month to month.
- **Example path:** `holdout/eval_prompts.jsonl`

### 3.5 `manifests/` — the ledgers
Two append-only [JSON Lines](https://jsonlines.org/) files (one JSON object per line).

**`dataset_manifest.jsonl`** — one row per ingested batch. Records provenance so you can
detect accidental duplicate ingestion and know precisely what each batch contained.

```json
{"batch": "2026-07", "ingested_at": "2026-07-02", "source": "dubai_tourism.md",
 "path": "01_training_input_data/raw/2026-07/dubai_tourism.md",
 "sha256": "976fa3bf…1800e37", "bytes": 104923, "words": 17490,
 "notes": "Initial seed batch: Dubai tourism overview."}
```

| Field | Meaning |
|-------|---------|
| `batch` | Month id (`YYYY-MM`) this data belongs to |
| `ingested_at` | Date the data was added |
| `source` / `path` | Original filename and its location under `raw/` |
| `sha256` | Content hash — lets you detect if the same file is ingested twice |
| `bytes` / `words` | Size metrics for tracking corpus growth over time |
| `notes` | Free-text description of the batch |

**`training_runs.jsonl`** — one row per training run. Records which batches the model saw
and how it scored, so any model version is fully traceable back to its data.

```json
{"run_id": "2026-08-01", "trained_at": "2026-08-01", "model_version": "v0.2.0",
 "strategy": "from_scratch", "batches": ["2026-07", "2026-08"],
 "checkpoint": "checkpoints/v0.2.0/", "eval": {"holdout_loss": 3.41}}
```

| Field | Meaning |
|-------|---------|
| `run_id` | Unique id for the run |
| `model_version` | Version tag produced by this run |
| `strategy` | `from_scratch` (train on all batches) or `continued` (resume checkpoint on newest batch) |
| `batches` | Exact list of batches this run trained on — the key provenance link |
| `checkpoint` | Where the resulting weights were saved |
| `eval.holdout_loss` | Score on `holdout/`, comparable across months |

### 3.6 `new_month.sh` — batch scaffolder
Creates the `raw/`, `interim/`, and `processed/` sub-folders for a new month so you don't
create them by hand.

```bash
./01_training_input_data/new_month.sh 2026-08   # or no argument → uses the current month
```

---

## 4. Worked example — adding August 2026 and retraining

Suppose it's August and you've collected two new documents: `dubai_events_2026.md` and
`uae_visa_rules.md`. Here is the full monthly cycle.

**Step 1 — Scaffold the batch**
```bash
./01_training_input_data/new_month.sh 2026-08
```
This creates `raw/2026-08/`, `interim/2026-08/`, and `processed/2026-08/`.

**Step 2 — Drop in the raw sources** (immutable from here on)
```
01_training_input_data/raw/2026-08/
├── dubai_events_2026.md
└── uae_visa_rules.md
```

**Step 3 — Clean into `interim/`** (your cleaning script's output)
```
01_training_input_data/interim/2026-08/
├── dubai_events_2026.txt
└── uae_visa_rules.txt
```

**Step 4 — Tokenize into `processed/`**
```
01_training_input_data/processed/2026-08/
├── train_000.bin
└── train_001.bin
```

**Step 5 — Record the batch** in `manifests/dataset_manifest.jsonl` (append two lines):
```json
{"batch": "2026-08", "ingested_at": "2026-08-01", "source": "dubai_events_2026.md", "path": "01_training_input_data/raw/2026-08/dubai_events_2026.md", "sha256": "…", "bytes": 51234, "words": 8900, "notes": "Events & festivals calendar."}
{"batch": "2026-08", "ingested_at": "2026-08-01", "source": "uae_visa_rules.md", "path": "01_training_input_data/raw/2026-08/uae_visa_rules.md", "sha256": "…", "bytes": 38210, "words": 6100, "notes": "Updated visa/entry rules."}
```

**Step 6 — Train.** Pick a strategy:
- *Full retrain from scratch:* feed the trainer `processed/2026-07/` **and** `processed/2026-08/`.
- *Continued training:* resume last month's checkpoint and train on just `processed/2026-08/`.

**Step 7 — Evaluate & log the run** in `manifests/training_runs.jsonl`:
```json
{"run_id": "2026-08-01", "trained_at": "2026-08-01", "model_version": "v0.2.0", "strategy": "from_scratch", "batches": ["2026-07", "2026-08"], "checkpoint": "checkpoints/v0.2.0/", "eval": {"holdout_loss": 3.41}}
```

**Result:** the folder now looks like this, and every artifact is traceable:
```
01_training_input_data/
├── raw/
│   ├── 2026-07/dubai_tourism.md
│   └── 2026-08/{dubai_events_2026.md, uae_visa_rules.md}
├── interim/
│   ├── 2026-07/…
│   └── 2026-08/…
├── processed/
│   ├── 2026-07/…
│   └── 2026-08/…
├── holdout/…
└── manifests/
    ├── dataset_manifest.jsonl   # 3 rows: 1 from July, 2 from August
    └── training_runs.jsonl      # 1 row: v0.2.0 trained on [2026-07, 2026-08]
```

Next month you repeat the cycle for `2026-09`. The corpus grows additively, and the
manifests remain a complete, auditable history of what the model learned and when.

---

## 5. Golden rules (quick reference)

1. **`raw/` is immutable** — never edit in place; add new files/versions instead.
2. **Back up `raw/` only** — `interim/` and `processed/` are always regenerable.
3. **Never train on `holdout/`** and never let it change, or your month-to-month
   comparisons become meaningless.
4. **One line per batch** in `dataset_manifest.jsonl`, **one line per run** in
   `training_runs.jsonl` — these ledgers are what make the whole thing reproducible.
5. **Batch id = `YYYY-MM`**, reused across `raw/`, `interim/`, and `processed/`.
