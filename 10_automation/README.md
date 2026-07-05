# Monthly Automation Phase

Chains every phase into a single command so a month of data becomes an evaluated model
checkpoint in one step. This is what makes "retrain every month" a routine rather than a
manual sequence.

## What it runs (in order)
```
raw/<batch>  ──▶ clean ──▶ tokenize ──▶ train | fine-tune ──▶ evaluate ──▶ checkpoint + logs
```
1. **(0) extract tokenizer** — once, from the Ollama GGUF, if not already done
2. **clean** — `02_tokenization/scripts/clean_batch.py`
3. **tokenize** — `02_tokenization/scripts/tokenize_batch.py`
4. **train** — from scratch (`training/`) or **continue** the latest checkpoint (`finetuning/`)
5. **evaluate** — `08_evaluation/scripts/evaluate.py` (perplexity + sample)

Each phase runs in its own venv; the orchestrator just calls them in order and tees a log to
`10_automation/logs/<batch>.log`.

## Run
```bash
./10_automation/scripts/run_month.sh 2026-08            # train from scratch on 2026-08
./10_automation/scripts/run_month.sh 2026-08 continue   # continue the latest checkpoint
./10_automation/scripts/run_month.sh                    # current month, from scratch
```
Prerequisite: this month's source documents are in `01_training_input_data/raw/<batch>/`
(use `01_training_input_data/new_month.sh <batch>` to scaffold the folders).

## Scheduling it monthly
Run it on the 1st of each month with cron (continue-mode incremental retraining):
```cron
0 3 1 * *  cd /Users/sandeeppandita/Desktop/llm_stepbystep-cicd && \
           ./10_automation/scripts/run_month.sh "$(date +\%Y-\%m)" continue >> 10_automation/logs/cron.log 2>&1
```
On macOS you can equivalently use a `launchd` plist, or Claude Code's `/schedule` for a
managed cron agent.

## Layout
```
automation/
├── scripts/run_month.sh       # the orchestrator
├── logs/<batch>.log           # per-month run log (created on run)
└── tests/test_automation.py   # validates the script + that every phase entrypoint exists
```

## Full-retrain vs. continue
- **scratch:** trains on the given batch from random init — a clean, reproducible model.
- **continue:** loads the most recent `checkpoints/*/model.npz` and adapts it to the new
  batch (faster, retains prior learning). Both log to
  `01_training_input_data/manifests/training_runs.jsonl`, so every model version stays traceable
  to the data and strategy that produced it.
