# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch GPT-style LLM built **in pure NumPy**, structured as a **monthly-retrainable
pipeline**. Folders are numbered `01_`–`11_` in execution order; each is a self-contained
phase with its own `config/`, `src/`, `scripts/`, `tests/`, and **its own virtualenv**
(`<phase>/.venv`). The trained model is exported to GGUF and published to Ollama as
`dubai-tourism-llm`. (GitHub repo name: `build-llm-stepbystep-with-ci-cd`.)

Phases: `01_training_input_data` (data store) → `02_tokenization` → `03_embeddings` →
`04_attention` → `05_mlp` → `06_transformer_block` → `07_training` → `08_evaluation` →
`09_finetuning` → `10_deployment` → `11_automation` (orchestrator).

## Critical constraint: NumPy only, no PyTorch

The dev machine is an **Intel Mac (x86_64) on Python 3.13**, where **PyTorch has no wheels**.
Everything — forward pass, hand-written backprop, AdamW — is NumPy on CPU. Do not add a
`torch` dependency; it will not install here. This is why the model is small and training is
slow (~seconds/step). Porting to GPU/PyTorch is a future step; the tokenized `tokens.bin`
files carry over unchanged.

## How the code is wired (non-obvious)

- **Per-phase venvs, not one environment.** Always invoke a phase's tools with its own venv:
  `<phase>/.venv/bin/python …`. Dep sets differ: most phases have `numpy pyyaml`;
  `02_tokenization` adds `tokenizers gguf`, `08_evaluation` adds `tokenizers`,
  `10_deployment` adds `gguf`. `11_automation` tests run with system `python3`.
- **Cross-phase imports use `sys.path`, not installed packages.** Scripts do
  `sys.path.insert(0, str(REPO / "06_transformer_block" / "src"))` and then
  `from model import GPTModel`. Modules are imported by bare filename (`model`,
  `input_embedding`, `attention`, `mlp`, `layer_norm`), and `REPO` is derived via
  `Path(__file__).resolve().parents[N]`. **Renaming a phase folder therefore breaks these
  hardcoded path strings** — grep for the old name across `.py/.sh/.yaml/.md` and update all
  references (this is how the earlier renumbering was done).
- **The model is assembled from the component phases.** `06_transformer_block/src/model.py`
  (`GPTModel`) imports `InputEmbedding` (03), `CausalMultiHeadAttention` (04), `FeedForward`
  (05), and `LayerNorm` (06) and stacks them. Each component implements `forward()`,
  `backward()`, `parameters()`, and `grads`/`gradients()`. `GPTModel.loss_and_backward()`
  runs the full chain; `07_training/src/optimizer.py` (`AdamW`) updates params **in place**
  (preserving array references). The backprop is verified by a numerical gradient check
  (~1e-6 in float64) and an overfit-to-zero test in `07_training/tests/`.
- **Config-driven; `vocab_size` is never hardcoded.** It is read at runtime from the batch's
  `01_training_input_data/processed/<batch>/meta.json` so it always matches the tokenizer.
- **The tokenizer is borrowed, not trained.** `02_tokenization` reads the
  **nomic-embed-text** WordPiece vocab directly from its local **Ollama GGUF blob**
  (`~/.ollama/models/...`), because the Ollama HTTP server cannot return token IDs. The
  extracted tokenizer is committed at `02_tokenization/vocab/nomic-embed-text/`. The token +
  positional **embeddings** (in `03_embeddings`) ARE trained from scratch.
- **Data flow & ledger.** Batches are dated `YYYY-MM`. Data moves
  `raw/ → interim/ → processed/` under `01_training_input_data/`; only `raw/` is the source
  of truth. Runs are logged as JSONL records in
  `01_training_input_data/manifests/{dataset_manifest,training_runs}.jsonl` (ingestion,
  tokenization, training, evaluation, deployment) — treat these as the pipeline's audit log.
- **Deployment = NumPy → GGUF → Ollama.** `10_deployment/src/export_gguf.py` maps the model
  onto llama.cpp's `gpt2` architecture: Q/K/V are **fused** into `attn_qkv`, linear weights
  are **transposed** to ggml's `(in,out)` layout, and the nomic WordPiece tokenizer is copied
  in. Then `deploy.py` writes a Modelfile and runs `ollama create`. `num_ctx` in
  `10_deployment/config/deploy.yaml` must equal the model's `context_length`.
- **Gitignored, not in the repo:** all `.venv/`, `checkpoints/` (the `model.npz`/`model.gguf`
  artifacts), `__pycache__`, and `*/logs/*.log`.

## Common commands

```bash
# One-time setup of a phase's venv (numpy pyyaml; add tokenizers for 02/08, gguf for 02/10)
python3 -m venv <phase>/.venv && <phase>/.venv/bin/pip install numpy pyyaml

# Run a single phase's tests
07_training/.venv/bin/python 07_training/tests/test_training.py

# Full test sweep (each phase with its own venv; 11_automation uses system python3)
for p in 02_tokenization 03_embeddings 04_attention 05_mlp 06_transformer_block \
         07_training 08_evaluation 09_finetuning 10_deployment; do
  $p/.venv/bin/python $p/tests/test_*.py; done
python3 11_automation/tests/test_automation.py

# End-to-end for one month (clean → tokenize → train|continue → evaluate → deploy)
./11_automation/scripts/run_month.sh 2026-07 scratch      # or: continue

# Individual phases
02_tokenization/.venv/bin/python 02_tokenization/scripts/extract_vocab.py            # once
02_tokenization/.venv/bin/python 02_tokenization/scripts/tokenize_batch.py 2026-07
07_training/.venv/bin/python 07_training/scripts/train.py 2026-07 --version v0.1.0 [--steps N]
09_finetuning/.venv/bin/python 09_finetuning/scripts/finetune.py --from checkpoints/v0.1.0/model.npz --batch 2026-08 --version v0.2.0
08_evaluation/.venv/bin/python 08_evaluation/scripts/evaluate.py --checkpoint checkpoints/v0.1.0/model.npz --batch 2026-07
10_deployment/.venv/bin/python 10_deployment/scripts/deploy.py --checkpoint checkpoints/v0.1.0/model.npz --version v0.1.0
```

Training hyperparameters live in `07_training/config/train.yaml` (model size + steps). A
long/overnight run should be wrapped in `caffeinate -i` so the Mac doesn't idle-sleep, and
launched in the background.

## Known state / gotchas

- **Tiny single-document corpus** (`01_training_input_data/raw/2026-07/dubai_tourism.md`,
  ~20k tokens) → models **overfit** readily (train loss → ~0, val loss rises). Only ~9% of
  the 30,522 token-embedding rows are ever trained. The bottleneck is data, not model size.
- The trainer **saves only the final-step checkpoint** (no best-val/early-stopping yet), so
  long runs can deploy an overfit model.
- **CI/CD is planned but not built.** The chosen tool is **Jenkins**, to run natively via
  Homebrew (no Docker). The full plan (Jenkinsfile + `12_cicd/` + a `pipeline-troubleshooter`
  subagent) is in `CICD_TROUBLESHOOTING_PLAN.md` and is currently on hold.
