# Troubleshooting / RCA Agent (LangGraph + local llama3.1)

A standalone local agent (like `12_cicd`) that **auto-scans the whole project and produces a
root-cause analysis** — not just symptoms. `llama3.1:8b` gathers evidence via tools and
synthesizes a `SYMPTOM / EVIDENCE / ROOT CAUSE / FIX / CONFIDENCE` report.

Design note: because an 8B model is a weak reasoner, **the tools do the detection** (return
`OK`/`PROBLEM` flags); the LLM's job is synthesis. That keeps the RCA reliable.

## What it inspects
- **`check_environment`** — Ollama reachable? required models present (`llama3.1:8b`,
  `nomic-embed-text`, `dubai-tourism-llm`)? per-phase `.venv`s? tokenizer vocab?
- **`scan_logs`** — recent pipeline / CI / phase logs for `ERROR|FAIL|Traceback`.
- **`analyze_manifests`** — training/eval/deploy metrics; flags **overfitting** (val ≫ train),
  high perplexity, or a train-without-deploy gap.
- **`run_tests`** — runs `12_cicd/run_all_tests.sh` to reproduce a code failure.
- **`read_source`** — inspects the implicated file/config.

## Run
```bash
PY=13_troubleshooting/.venv/bin/python
$PY 13_troubleshooting/scripts/troubleshoot.py            # llama3.1 writes the RCA report
$PY 13_troubleshooting/scripts/troubleshoot.py --no-llm   # raw evidence report (fast, no LLM)
$PY 13_troubleshooting/scripts/troubleshoot.py --no-llm --run-tests
$PY 13_troubleshooting/tests/test_troubleshooter.py       # tool tests
```
Reports are written to `13_troubleshooting/logs/rca_<timestamp>.log`.

## Setup
```bash
python3 -m venv 13_troubleshooting/.venv
13_troubleshooting/.venv/bin/pip install langgraph langchain-core langchain-ollama pyyaml
```
Requires Ollama running with `llama3.1:8b`.

## Notes
- `--no-llm` is a deterministic project **health check** (env + logs + manifest anomalies) with
  no LLM — reliable and fast; the default mode adds llama3.1's RCA synthesis on top.
- The agent's system prompt embeds this project's **failure catalog** (from `CLAUDE.md`), so it
  maps symptoms to causes: import errors → venv/rename; missing model → Ollama; `val ≫ train` /
  high perplexity → overfitting on the tiny corpus; GGUF create failure → export layout.
- llama3.1:8b on CPU is slow (minutes per run); switch `model:` to `llama3.2:3b` for speed.
