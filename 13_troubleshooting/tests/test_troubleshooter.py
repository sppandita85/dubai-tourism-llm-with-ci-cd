#!/usr/bin/env python
"""Troubleshooter tool tests — deterministic, no LLM required.

    python 13_troubleshooting/tests/test_troubleshooter.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import tools as T  # noqa: E402


def main() -> int:
    failures = []

    # 1. check_environment returns OK/PROBLEM flags for models + venvs + vocab.
    env = T.check_environment(["llama3.1:8b", "dubai-tourism-llm"], "http://localhost:11434")
    if "ENVIRONMENT:" not in env or not any(f in env for f in ("OK", "PROBLEM")):
        failures.append(f"check_environment malformed: {env[:80]}")
    # unreachable host -> graceful PROBLEM
    bad = T.check_environment(["x"], "http://localhost:1")
    if "PROBLEM" not in bad:
        failures.append("check_environment should flag an unreachable Ollama host")

    # 2. analyze_manifests flags overfitting from a synthetic manifest.
    with tempfile.TemporaryDirectory() as td:
        mf = Path(td) / "training_runs.jsonl"
        mf.write_text("\n".join(json.dumps(r) for r in [
            {"strategy": "from_scratch", "model_version": "v1",
             "eval": {"final_train_loss": 0.12, "final_val_loss": 9.85}},
            {"record_type": "evaluation", "eval": {"perplexity": 19008, "holdout_loss": 9.85}},
            {"record_type": "deployment", "version": "v1"},
        ]))
        # analyze_manifests resolves against REPO; point it via a relative path inside a temp
        # repo-like dir by temporarily using an absolute path trick: copy logic through REPO.
        import importlib
        rel = mf  # absolute path; patch REPO for this call
        saved = T.REPO
        try:
            T.REPO = Path(td)
            res = T.analyze_manifests("training_runs.jsonl")
        finally:
            T.REPO = saved
        if "overfitting" not in res.lower():
            failures.append(f"analyze_manifests did not flag overfitting: {res}")
        if "perplexity" not in res.lower():
            failures.append("analyze_manifests did not report perplexity")

    # 3. scan_logs handles the real repo (returns a string, no crash).
    logs = T.scan_logs(["11_automation/logs/*.log", "12_cicd/logs/*.log"])
    if "LOGS" not in logs:
        failures.append(f"scan_logs malformed: {logs[:60]}")

    # 4. read_source reads a real file.
    src = T.read_source("13_troubleshooting/config/agent.yaml", "model")
    if "llama3.1" not in src:
        failures.append("read_source did not return file content")

    # 5. graph builds the 5 tools.
    try:
        from graph import build_tools
        names = {t.name for t in build_tools({
            "required_models": [], "ollama_host": "http://localhost:11434",
            "log_globs": [], "manifest": "x", "test_runner": "y"})}
        if names != {"check_environment", "scan_logs", "analyze_manifests",
                     "run_tests", "read_source"}:
            failures.append(f"unexpected tool names: {names}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"build_tools failed: {e}")

    if failures:
        print("TROUBLESHOOTER TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All troubleshooter tests passed.")
    print("  analyze_manifests correctly flags the overfitting signature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
