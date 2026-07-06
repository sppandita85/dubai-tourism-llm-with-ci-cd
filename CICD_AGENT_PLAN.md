# Plan: Local-LLM CI/CD agent — LangGraph + Ollama (llama3.1)

## Context
The project is an 11-phase NumPy LLM pipeline on GitHub
(`sppandita85/build-llm-stepbystep-with-ci-cd`, public) whose model is deployed to Ollama as
`dubai-tourism-llm`. The goal is a **CI/CD agent, built with LangGraph and powered by a local
Ollama model (`llama3.1:8b`)**, that on demand **pulls the code from GitHub, runs the test
suites, and smoke-tests the deployed Ollama model**, then emits a PASS/FAIL verdict.
Everything runs locally — LangGraph orchestrates, `ChatOllama` talks to your local Ollama, no
cloud. (Supersedes the earlier Jenkins plan; they can share `run_all_tests.sh`.)

Verified: `llama3.1:8b` supports tool-calling (`capabilities: [completion, tools]`), so
`ChatOllama(...).bind_tools(...)` works. Test venv deps = `numpy pyyaml tokenizers gguf`;
`10_deployment` self-skips without the nomic blob; `02_tokenization` roundtrip uses the
committed vocab.

## Framework
**LangGraph** (graph-based agent orchestration) + **langchain-ollama** (`ChatOllama`) + the
official tool primitives. No cloud, no OpenAI. Dependencies (in `12_cicd/.venv`):
`langgraph langchain-core langchain-ollama pyyaml`. A separate `12_cicd/.ci-venv`
(`numpy pyyaml tokenizers gguf`) built by `run_all_tests.sh` runs the phase tests.

## New phase: `12_cicd/`
```
12_cicd/
├── config/agent.yaml       # model, repo url/branch, ollama test model, workspace, recursion_limit
├── src/
│   ├── tools.py            # @tool functions: git_sync, run_tests, check_ollama_model, read_manifest
│   └── graph.py            # builds the LangGraph agent (ChatOllama + tools)
├── scripts/run_ci_agent.py # entrypoint: invoke the graph (or --no-llm deterministic fallback)
├── run_all_tests.sh        # shared test runner (build/reuse .ci-venv, run every phase test)
├── tests/test_ci_agent.py  # tests the @tool functions deterministically (no LLM needed)
├── README.md
└── .gitignore              # .venv/ .ci-venv/ workspace/ logs/
```

## The tools (`src/tools.py`) — LangChain `@tool`s, each returns a short string
- `git_sync()` — shallow clone / fetch+reset the repo/branch into `workspace/`; returns commit hash + subject.
- `run_tests()` — runs the LOCAL `run_all_tests.sh <workspace>`; returns pass/fail counts, failing suite names, log path.
- `check_ollama_model()` — calls `dubai-tourism-llm` via Ollama; returns the generated text (truncated).
- `read_manifest()` — last few `training_runs.jsonl` records from the workspace.

## How it executes with LangGraph

### Primary: a prebuilt ReAct agent (llama3.1 drives the tools)
```python
# src/graph.py
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tools import git_sync, run_tests, check_ollama_model, read_manifest

def build_agent(cfg):
    llm = ChatOllama(model=cfg["model"], base_url=cfg["ollama_host"], temperature=0)
    return create_react_agent(llm, tools=[git_sync, run_tests, check_ollama_model, read_manifest])
```
```python
# scripts/run_ci_agent.py (core)
agent = build_agent(cfg)
result = agent.invoke(
    {"messages": [("system", CI_SYSTEM_PROMPT), ("user", "Run CI on the latest build.")]},
    config={"recursion_limit": cfg["recursion_limit"]},
)
print(result["messages"][-1].content)   # the PASS/FAIL verdict
```
LangGraph's ReAct graph is `agent → tools → agent → …`: llama3.1 reads the system prompt,
calls `git_sync`, then `run_tests`, then `check_ollama_model` (the built-in `ToolNode` executes
each and feeds results back), and finally returns a verdict when it stops calling tools.
`CI_SYSTEM_PROMPT` pins the job and the required `PASS`/`FAIL` output format.

### Optional enhancement: a custom `StateGraph` with a fail-fast edge
For deterministic control (don't bother smoke-testing the model if tests already failed):
```
        ┌──────────┐   ┌───────────┐   tests failed   ┌─────────┐
START ─▶│ git_sync │─▶│ run_tests  │───────────────▶ │ verdict │─▶ END
        └──────────┘   └───────────┘                  └─────────┘
                            │ tests passed                 ▲
                            ▼                               │
                       ┌──────────────┐                    │
                       │ check_model  │────────────────────┘
                       └──────────────┘
```
```python
from langgraph.graph import StateGraph, START, END
g = StateGraph(CIState)                 # CIState: commit, test_summary, passed, model_output, verdict
g.add_node("git_sync", sync_node); g.add_node("run_tests", test_node)
g.add_node("check_model", model_node); g.add_node("verdict", verdict_node)   # verdict_node uses llama3.1 to summarize
g.add_edge(START, "git_sync"); g.add_edge("git_sync", "run_tests")
g.add_conditional_edges("run_tests", lambda s: "check_model" if s["passed"] else "verdict")
g.add_edge("check_model", "verdict"); g.add_edge("verdict", END)
app = g.compile()                       # optional: checkpointer= for resumable runs
```
Here the tool nodes run deterministically and llama3.1 is used in `verdict_node` to write the
human-readable PASS/FAIL summary. This is the more reliable option given llama3.1:8b's
tool-calling can be inconsistent.

## How to run it
```bash
# one-time
python3 -m venv 12_cicd/.venv
12_cicd/.venv/bin/pip install langgraph langchain-core langchain-ollama pyyaml

# run the agent (pulls from GitHub, tests, smoke-tests the model, prints PASS/FAIL)
12_cicd/.venv/bin/python 12_cicd/scripts/run_ci_agent.py

# deterministic, no-LLM run of the same steps (fast, reliable)
12_cicd/.venv/bin/python 12_cicd/scripts/run_ci_agent.py --no-llm
```
Runtime: clone → run the 10 phase suites on the freshly-pulled build via `.ci-venv` → query
`dubai-tourism-llm` → verdict. Transcript + verdict saved to `12_cicd/logs/ci_run_<ts>.log`.
(llama3.1:8b tool-calling on CPU is slow — a full agentic run takes several minutes.)

## Config (`config/agent.yaml`)
`model: llama3.1:8b`, `ollama_host: http://localhost:11434`,
`repo_url: https://github.com/sppandita85/build-llm-stepbystep-with-ci-cd.git`, `branch: main`,
`workspace: 12_cicd/workspace`, `test_model: dubai-tourism-llm`, `test_prompt: "Dubai is"`,
`recursion_limit: 25`.

## Root `README.md`
Add a short **CI/CD agent (LangGraph + llama3.1)** section: what it does, how to run it, deps.

## Verification
1. `12_cicd/.venv/bin/python 12_cicd/tests/test_ci_agent.py` → the `@tool` functions work: a
   shallow clone succeeds, `run_tests` reports the suites, `check_ollama_model` returns text,
   `read_manifest` returns records (graceful skips if offline).
2. **Deterministic e2e:** `run_ci_agent.py --no-llm` → clones from GitHub, runs all 10 suites
   (pass; deployment self-skips if needed), smoke-tests `dubai-tourism-llm`, prints PASS.
3. **Agentic e2e (LangGraph):** `run_ci_agent.py` → llama3.1 drives the ReAct graph; confirm
   the log shows real tool calls (`git_sync` → `run_tests` → `check_ollama_model`) and a
   correct PASS/FAIL verdict.
4. Commit + push `12_cicd/` so a fresh clone contains the agent + runner.

---
*Status: awaiting review. Not yet implemented. To proceed, say "build it" (or adjust first).*
