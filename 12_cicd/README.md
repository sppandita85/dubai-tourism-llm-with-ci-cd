# CI/CD Agent (LangGraph + local llama3.1)

An agentic CI/CD runner whose **brain is a local Ollama model (`llama3.1:8b`)**, orchestrated
with **LangGraph**. On demand it **pulls the code from GitHub, runs every phase's test suite,
smoke-tests the deployed Ollama model (`dubai-tourism-llm`), and reports a PASS/FAIL verdict** —
all locally, no cloud.

## How it works
`llama3.1` runs as a LangGraph ReAct agent (`agent → tools → agent → …`) and calls four tools
until it can conclude:

```
git_sync ──▶ run_tests ──▶ check_model ──▶  VERDICT: PASS | FAIL
 (GitHub)    (10 suites)   (Ollama model)
```

- **`git_sync`** — shallow-clone/refresh the repo into `12_cicd/workspace/`.
- **`run_tests`** — `run_all_tests.sh` runs every `NN_phase/tests/test_*.py` against that
  checkout, in a shared `.ci-venv` (`numpy pyyaml tokenizers gguf`).
- **`check_model`** — generate from `dubai-tourism-llm` via Ollama to prove it responds.
- **`read_manifest`** — recent training/eval/deploy records for context.

## Layout
```
12_cicd/
├── config/agent.yaml       # model, repo url/branch, workspace, test model/prompt
├── src/tools.py            # the 4 tools (plain Python: git, subprocess, urllib)
├── src/graph.py            # LangGraph ReAct agent (ChatOllama + StructuredTools)
├── scripts/run_ci_agent.py # entrypoint (agentic, or --no-llm deterministic)
├── run_all_tests.sh        # shared phase-test runner (reusable by Jenkins too)
└── tests/test_ci_agent.py  # deterministic tool tests (no LLM needed)
```

## Setup
```bash
python3 -m venv 12_cicd/.venv
12_cicd/.venv/bin/pip install langgraph langchain-core langchain-ollama pyyaml
```
Requires a running Ollama with `llama3.1:8b` pulled (the agent's brain) and
`dubai-tourism-llm` deployed (the model under test).

## Run
```bash
PY=12_cicd/.venv/bin/python
$PY 12_cicd/scripts/run_ci_agent.py            # llama3.1 drives the tools (slow on CPU: minutes)
$PY 12_cicd/scripts/run_ci_agent.py --no-llm   # same steps, deterministic + fast (no LLM)
$PY 12_cicd/tests/test_ci_agent.py             # tool sanity tests
```
Exit code is `0` on PASS, `1` on FAIL. A transcript + verdict is written to
`12_cicd/logs/ci_run_<timestamp>.log`.

## Notes
- **`--no-llm`** runs the exact same git→test→model→verdict sequence without the LLM. Use it
  for reliable CI (and to sanity-check the machinery); use the default agentic mode to watch
  `llama3.1` reason and call the tools.
- The agent tests **committed code on `main`**, not local uncommitted changes.
- `llama3.1:8b` tool-calling on CPU is slow (~30–60s per step); switch `model:` in
  `agent.yaml` to `llama3.2:3b` for a faster (slightly weaker) loop.
- The runner and the `.ci-venv` live locally; a fresh clone only needs the phase code +
  committed tokenizer vocab (deployment test self-skips if the nomic blob is absent).
