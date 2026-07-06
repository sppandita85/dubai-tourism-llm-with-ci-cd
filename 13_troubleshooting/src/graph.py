"""LangGraph RCA agent: llama3.1 synthesizes tool findings into a root-cause analysis.

The tools do the detection (returning OK/PROBLEM flags); the model's job is to gather that
evidence and write a structured RCA report. Config-bound, zero-arg tools keep the 8B model's
tool-calling reliable.
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools as T  # noqa: E402

RCA_SYSTEM_PROMPT = """You are a root-cause-analysis (RCA) agent for an 11-phase NumPy LLM
pipeline with local Ollama deployment and CI. Gather evidence with the tools, then explain the
ROOT CAUSE of any problem — do not just restate symptoms.

You MUST call ALL THREE evidence tools below before you conclude anything — in this order,
each exactly once — EVEN IF an earlier tool reports everything OK. A clean environment does
NOT mean the project is healthy: training-quality problems appear ONLY in analyze_manifests.
Do not write any verdict until all three have been called.
1. check_environment  - Ollama, required models, per-phase venvs, tokenizer vocab.
2. scan_logs          - recent pipeline/CI logs for errors and tracebacks.
3. analyze_manifests  - training/eval/deploy metrics and anomalies (overfitting, perplexity).
Only after all three: if a code failure was found, call run_tests to reproduce it and
read_source to inspect the file. Then STOP calling tools. Report the FIRST PROBLEM found across
all evidence (a PROBLEM line in any tool output is a real issue you must explain).

Known failure catalog for this project (map symptoms -> root cause):
- ModuleNotFoundError / import error  -> a phase .venv is missing/broken, or a folder was
  renamed and a sys.path/import reference wasn't updated.
- "ollama unreachable" or a missing model -> Ollama not running or the model wasn't pulled/deployed.
- missing tokenizer.json -> vocab not extracted; run 02_tokenization extract_vocab.py.
- val_loss >> train_loss, or perplexity in the thousands -> OVERFITTING on the tiny single-
  document corpus (data problem, not a code bug); fix with more data / early stopping / smaller model.
- GGUF 'ollama create' failure -> tensor layout or architecture mismatch in 10_deployment export.

Output EXACTLY this report and nothing after it:
SYMPTOM: <one line>
EVIDENCE: <key findings from the tools>
ROOT CAUSE: <the underlying cause, not the symptom>
RECOMMENDED FIX: <concrete next step>
CONFIDENCE: <high|medium|low>
If everything is healthy, say so with ROOT CAUSE: none."""


def build_tools(cfg: dict) -> list[StructuredTool]:
    def check_environment() -> str:
        """Check Ollama, required models, per-phase venvs, and tokenizer vocab."""
        return T.check_environment(cfg["required_models"], cfg["ollama_host"])

    def scan_logs() -> str:
        """Scan recent pipeline/CI logs for errors, failures, and tracebacks."""
        return T.scan_logs(cfg["log_globs"])

    def analyze_manifests() -> str:
        """Analyze training/eval/deploy manifest records and flag anomalies."""
        return T.analyze_manifests(cfg["manifest"])

    def run_tests() -> str:
        """Run the phase test suites to reproduce/confirm a code failure (slow)."""
        return T.run_tests(cfg["test_runner"])

    def read_source(path_or_glob: str, pattern: str = "") -> str:
        """Read a bounded slice of a repo file (optionally only lines containing `pattern`)."""
        return T.read_source(path_or_glob, pattern or None)

    return [StructuredTool.from_function(f) for f in
            (check_environment, scan_logs, analyze_manifests, run_tests, read_source)]


def build_agent(cfg: dict):
    llm = ChatOllama(model=cfg["model"], base_url=cfg["ollama_host"], temperature=0)
    return create_react_agent(llm, tools=build_tools(cfg))
