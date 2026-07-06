"""LangGraph CI agent: a ReAct agent driven by a local Ollama model (llama3.1).

`build_agent(cfg)` returns a compiled LangGraph agent whose tools wrap the plain functions in
tools.py, bound to the config so the LLM calls them with no arguments (more reliable for an
8B model). The agent loop is LangGraph's prebuilt `agent -> tools -> agent -> ...` until the
model stops calling tools and emits a verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools as T  # noqa: E402

CI_SYSTEM_PROMPT = """You are a CI/CD agent for an LLM pipeline repository.
Use the tools in this exact order, each at most once:
1. git_sync  - pull the latest code from GitHub.
2. run_tests - run all phase test suites on that code.
3. check_model - confirm the deployed Ollama model still responds.
Then STOP calling tools and reply with a final answer whose first line is exactly:
VERDICT: PASS   (only if run_tests passed AND check_model responded)
or
VERDICT: FAIL   (otherwise)
Follow it with 1-2 sentences citing the tool results. Rely only on tool outputs; never invent
results."""


def build_tools(cfg: dict) -> list[StructuredTool]:
    def git_sync() -> str:
        """Pull the latest code from the GitHub repo into the local workspace."""
        return T.git_sync(cfg["repo_url"], cfg["branch"], cfg["workspace"])

    def run_tests() -> str:
        """Run all phase test suites against the pulled code; returns a pass/fail summary."""
        return T.run_tests(cfg["workspace"], cfg["runner"])

    def check_model() -> str:
        """Query the deployed Ollama model to confirm it responds."""
        return T.check_ollama_model(cfg["test_model"], cfg["test_prompt"], cfg["ollama_host"])

    def read_manifest() -> str:
        """Read recent training/eval/deploy records from the pipeline manifest."""
        return T.read_manifest(cfg["workspace"])

    return [StructuredTool.from_function(f)
            for f in (git_sync, run_tests, check_model, read_manifest)]


def build_agent(cfg: dict):
    llm = ChatOllama(model=cfg["model"], base_url=cfg["ollama_host"], temperature=0)
    return create_react_agent(llm, tools=build_tools(cfg))
