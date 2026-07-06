#!/usr/bin/env python
"""CI-agent tests: exercise the tool functions deterministically (no LLM required).

Kept lightweight — does NOT run the full phase suite (that's the e2e job). Verifies the
tools behave and degrade gracefully when Ollama/network are unavailable.

    python 12_cicd/tests/test_ci_agent.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import tools as T  # noqa: E402


def main() -> int:
    failures = []

    # 1. run_all_tests.sh exists, is executable, valid bash.
    runner = ROOT / "run_all_tests.sh"
    if not runner.exists():
        failures.append("run_all_tests.sh missing")
    elif subprocess.run(["bash", "-n", str(runner)]).returncode != 0:
        failures.append("run_all_tests.sh has bash syntax errors")

    # 2. read_manifest returns records from the local repo's manifest.
    man = T.read_manifest(str(REPO))
    if "recent runs" not in man and "no manifest" not in man:
        failures.append(f"read_manifest unexpected output: {man[:80]}")

    # 3. check_ollama_model: reachable model -> "MODEL OK"; unreachable -> graceful FAILED.
    ok = T.check_ollama_model("dubai-tourism-llm", "Dubai is", "http://localhost:11434")
    if not (ok.startswith("MODEL OK") or ok.startswith("MODEL CHECK FAILED")):
        failures.append(f"check_ollama_model malformed: {ok[:80]}")
    bad = T.check_ollama_model("dubai-tourism-llm", "hi", "http://localhost:1")  # bad port
    if not bad.startswith("MODEL CHECK FAILED"):
        failures.append("check_ollama_model should fail gracefully on a bad host")

    # 4. graph.py builds tools + agent object (imports LangGraph + ChatOllama).
    try:
        from graph import build_tools, build_agent  # noqa: F401
        tools = build_tools({"repo_url": "x", "branch": "main", "workspace": "w",
                             "runner": "r", "test_model": "m", "test_prompt": "p",
                             "ollama_host": "http://localhost:11434"})
        names = {t.name for t in tools}
        if names != {"git_sync", "run_tests", "check_model", "read_manifest"}:
            failures.append(f"unexpected tool names: {names}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"graph.build_tools failed: {e}")

    if failures:
        print("CI-AGENT TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All CI-agent tests passed.")
    print(f"  model check: {ok[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
