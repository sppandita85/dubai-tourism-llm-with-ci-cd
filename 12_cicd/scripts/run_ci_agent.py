#!/usr/bin/env python
"""Run the CI/CD agent: pull from GitHub -> run tests -> smoke-test the Ollama model -> verdict.

    python 12_cicd/scripts/run_ci_agent.py            # LangGraph + llama3.1 drives the tools
    python 12_cicd/scripts/run_ci_agent.py --no-llm   # deterministic: run the steps directly

Writes a transcript + verdict to 12_cicd/logs/ci_run_<timestamp>.log.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # 12_cicd/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import tools as T  # noqa: E402


def load_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text())
    # resolve paths to absolute against the repo root
    cfg["workspace"] = str((REPO / cfg["workspace"]).resolve())
    cfg["runner"] = str((ROOT / "run_all_tests.sh").resolve())
    return cfg


def log_path() -> Path:
    d = ROOT / "logs"
    d.mkdir(exist_ok=True)
    return d / f"ci_run_{datetime.now():%Y%m%d_%H%M%S}.log"


def run_no_llm(cfg: dict) -> tuple[str, str]:
    """Deterministic sequence, no LLM. Returns (verdict, transcript)."""
    steps = []
    steps.append(T.git_sync(cfg["repo_url"], cfg["branch"], cfg["workspace"]))
    tests = T.run_tests(cfg["workspace"], cfg["runner"]); steps.append(tests)
    passed = tests.startswith("TESTS PASSED")
    if passed:
        model = T.check_ollama_model(cfg["test_model"], cfg["test_prompt"], cfg["ollama_host"])
    else:
        model = "(skipped model check — tests failed)"
    steps.append(model)
    ok = passed and model.startswith("MODEL OK")
    verdict = f"VERDICT: {'PASS' if ok else 'FAIL'}"
    return verdict, "\n\n".join(steps) + "\n\n" + verdict


def run_agentic(cfg: dict) -> tuple[str, str]:
    """LangGraph ReAct agent driven by llama3.1. Returns (verdict, transcript)."""
    from graph import build_agent, CI_SYSTEM_PROMPT
    agent = build_agent(cfg)
    result = agent.invoke(
        {"messages": [("system", CI_SYSTEM_PROMPT), ("user", "Run CI on the latest build.")]},
        config={"recursion_limit": cfg["recursion_limit"]},
    )
    lines = []
    for m in result["messages"]:
        role = m.__class__.__name__.replace("Message", "").lower()
        for tc in (getattr(m, "tool_calls", None) or []):
            lines.append(f"[tool-call] {tc['name']}()")
        content = (m.content or "").strip()
        if content:
            lines.append(f"[{role}] {content}")
    final = (result["messages"][-1].content or "").strip()
    verdict = next((l for l in final.splitlines() if l.upper().startswith("VERDICT")), final)
    return verdict, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="run the steps deterministically without the LLM")
    args = ap.parse_args()
    cfg = load_cfg()

    mode = "deterministic (--no-llm)" if args.no_llm else f"LangGraph + {cfg['model']}"
    print(f"=== CI/CD agent — mode: {mode} ===")
    print(f"    repo: {cfg['repo_url']} @ {cfg['branch']}\n")

    verdict, transcript = run_no_llm(cfg) if args.no_llm else run_agentic(cfg)

    lp = log_path()
    lp.write_text(f"mode: {mode}\n\n{transcript}\n")
    print(transcript if args.no_llm else "")
    print(f"\n{'='*50}\n{verdict}\n{'='*50}")
    print(f"log: {lp.relative_to(REPO)}")
    return 0 if "PASS" in verdict.upper() else 1


if __name__ == "__main__":
    raise SystemExit(main())
