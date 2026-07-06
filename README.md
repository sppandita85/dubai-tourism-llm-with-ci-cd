# build-llm-stepbystep-with-ci-cd

Trains **`dubai-tourism-llm`** — a from-scratch LLM you can run in Ollama.

A **from-scratch LLM** built step by step in pure NumPy, organized as a **monthly-retrainable
pipeline**. Each stage is a self-contained, numbered phase with its own `config/`, `src/`,
`scripts/`, `tests/`, and virtual environment.

## Pipeline (folders are numbered in execution order)

| # | Phase | What it does |
|---|-------|--------------|
| 01 | `01_training_input_data/` | Dated data store: `raw → interim → processed`, holdout, manifests |
| 02 | `02_tokenization/` | Text → token IDs using the **nomic-embed-text** tokenizer read from its Ollama GGUF |
| 03 | `03_embeddings/` | Token + positional embeddings |
| 04 | `04_attention/` | Causal multi-head self-attention |
| 05 | `05_mlp/` | Position-wise feed-forward network |
| 06 | `06_transformer_block/` | LayerNorm + residuals + the assembled GPT model |
| 07 | `07_training/` | Backprop + AdamW + checkpointing → **the trained model** |
| 08 | `08_evaluation/` | Perplexity + text generation |
| 09 | `09_finetuning/` | Continue training from a checkpoint (monthly updates) |
| 10 | `10_deployment/` | Export checkpoint → GGUF and publish to **Ollama** (`ollama run`) |
| 11 | `11_automation/` | Monthly orchestrator: clean → tokenize → train → evaluate → deploy |

The trained model lands in `checkpoints/<version>/model.npz` (step 07 from scratch, step 09
for fine-tuned updates), and step 10 makes it runnable in Ollama as `dubai-tourism-llm:<version>`.

## Quickstart

Each phase has its own venv (`<phase>/.venv`). Typical one-time setup per phase:
```bash
python3 -m venv <phase>/.venv
<phase>/.venv/bin/pip install numpy pyyaml   # + tokenizers for 02 and 08
```

Run a single month end to end (assumes docs in `01_training_input_data/raw/<batch>/`):
```bash
./11_automation/scripts/run_month.sh 2026-08 continue
```

Or step by step:
```bash
02_tokenization/.venv/bin/python 02_tokenization/scripts/extract_vocab.py     # once
02_tokenization/.venv/bin/python 02_tokenization/scripts/clean_batch.py   2026-07
02_tokenization/.venv/bin/python 02_tokenization/scripts/tokenize_batch.py 2026-07
07_training/.venv/bin/python 07_training/scripts/train.py 2026-07 --version v0.1.0
08_evaluation/.venv/bin/python 08_evaluation/scripts/evaluate.py \
    --checkpoint checkpoints/v0.1.0/model.npz --batch 2026-07
10_deployment/.venv/bin/python 10_deployment/scripts/deploy.py \
    --checkpoint checkpoints/v0.1.0/model.npz --version v0.1.0    # -> ollama run dubai-tourism-llm:v0.1.0
```

## Monthly retraining

Drop next month's documents in `01_training_input_data/raw/2026-08/`, then run the
orchestrator in `continue` mode to fine-tune the latest checkpoint on the new data. Every
run is logged to `01_training_input_data/manifests/training_runs.jsonl`, so each model
version is traceable to the data and strategy that produced it.

## Correctness

The backward pass is verified two ways: a **numerical gradient check** (analytic vs.
finite-difference, ~1e-6 in float64) and an **overfit-to-zero test** (a small model memorizes
a fixed batch, loss `ln(vocab) → ~0.01`). A trained run on the sample corpus takes the loss
from **10.34 → 4.49**. Run any phase's tests with `<phase>/.venv/bin/python <phase>/tests/*.py`.

## CI/CD agent (LangGraph + local llama3.1)

`12_cicd/` is an **agentic CI runner** whose brain is a local Ollama model (`llama3.1:8b`),
orchestrated with **LangGraph**. It pulls the code from GitHub, runs all 10 phase test suites,
smoke-tests the deployed `dubai-tourism-llm` model, and returns a PASS/FAIL verdict — fully
local, no cloud.

```bash
12_cicd/.venv/bin/python 12_cicd/scripts/run_ci_agent.py            # llama3.1 drives the tools
12_cicd/.venv/bin/python 12_cicd/scripts/run_ci_agent.py --no-llm   # same steps, deterministic + fast
```
Flow: `git_sync → run_tests → check_model → VERDICT`. See [12_cicd/README.md](12_cicd/README.md).

`13_troubleshooting/` is a companion **root-cause-analysis agent** (also LangGraph + llama3.1):
it auto-scans the whole project — environment, logs, and JSONL manifests — and reports the
root cause of any problem (`SYMPTOM → EVIDENCE → ROOT CAUSE → FIX`). Its `--no-llm` mode is a
deterministic health check.
```bash
13_troubleshooting/.venv/bin/python 13_troubleshooting/scripts/troubleshoot.py            # RCA
13_troubleshooting/.venv/bin/python 13_troubleshooting/scripts/troubleshoot.py --no-llm   # health check
```
See [13_troubleshooting/README.md](13_troubleshooting/README.md).

## Note on scale

This is **pure NumPy on CPU** (the dev machine is an Intel Mac on Python 3.13, where PyTorch
has no wheels). It is correct and complete for learning and small monthly runs; for real
scale, port the components to PyTorch/GPU — the tokenized `tokens.bin` files carry over
unchanged.
