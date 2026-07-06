#!/usr/bin/env python
"""Troubleshooting / RCA agent: auto-scan the project and explain the root cause.

    python 13_troubleshooting/scripts/troubleshoot.py            # llama3.1 writes an RCA report
    python 13_troubleshooting/scripts/troubleshoot.py --no-llm   # raw evidence report (no LLM)
    python 13_troubleshooting/scripts/troubleshoot.py --no-llm --run-tests

Writes a transcript/report to 13_troubleshooting/logs/rca_<timestamp>.log.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # 13_troubleshooting/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import tools as T  # noqa: E402


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "agent.yaml").read_text())


def log_path() -> Path:
    d = ROOT / "logs"; d.mkdir(exist_ok=True)
    return d / f"rca_{datetime.now():%Y%m%d_%H%M%S}.log"


def evidence_report(cfg: dict, run_tests: bool) -> str:
    parts = [
        T.check_environment(cfg["required_models"], cfg["ollama_host"]),
        T.scan_logs(cfg["log_globs"]),
        T.analyze_manifests(cfg["manifest"]),
    ]
    if run_tests:
        parts.append(T.run_tests(cfg["test_runner"]))
    report = "\n\n".join(parts)
    problems = [l for l in report.splitlines() if "PROBLEM" in l]
    summary = ("PROBLEMS DETECTED:\n" + "\n".join(problems)) if problems \
        else "No problems detected — project looks healthy."
    return f"{report}\n\n{'='*54}\n{summary}"


def agentic_rca(cfg: dict) -> str:
    from graph import build_agent, RCA_SYSTEM_PROMPT
    agent = build_agent(cfg)
    result = agent.invoke(
        {"messages": [("system", RCA_SYSTEM_PROMPT),
                      ("user", "Troubleshoot the project and give the root-cause analysis.")]},
        config={"recursion_limit": cfg["recursion_limit"]},
    )
    lines = []
    for m in result["messages"]:
        for tc in (getattr(m, "tool_calls", None) or []):
            lines.append(f"[tool-call] {tc['name']}({tc.get('args') or ''})")
        c = (m.content or "").strip()
        if c:
            lines.append(f"[{m.__class__.__name__.replace('Message','').lower()}] {c}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="print raw evidence, no LLM synthesis")
    ap.add_argument("--run-tests", action="store_true", help="also run the test suites")
    args = ap.parse_args()
    cfg = load_cfg()

    mode = "deterministic evidence report" if args.no_llm else f"LangGraph RCA + {cfg['model']}"
    print(f"=== troubleshooting agent — {mode} ===\n")
    report = evidence_report(cfg, args.run_tests) if args.no_llm else agentic_rca(cfg)
    print(report)

    lp = log_path(); lp.write_text(f"mode: {mode}\n\n{report}\n")
    print(f"\nlog: {lp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
