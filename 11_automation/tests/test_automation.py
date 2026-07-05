#!/usr/bin/env python
"""Automation-phase checks: the orchestrator is valid and wires to real entrypoints.

    python 11_automation/tests/test_automation.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent


def main() -> int:
    failures = []
    script = ROOT / "scripts" / "run_month.sh"

    if not script.exists():
        print("FAIL: run_month.sh missing"); return 1
    # Executable bit set.
    if not (script.stat().st_mode & 0o111):
        failures.append("run_month.sh is not executable")
    # Valid bash syntax.
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        failures.append(f"bash -n failed: {r.stderr.strip()}")
    # Rejects a bad batch id.
    r = subprocess.run(["bash", str(script), "not-a-month"], capture_output=True, text=True)
    if r.returncode == 0:
        failures.append("script accepted an invalid batch id")

    # Every phase entrypoint the orchestrator calls must exist.
    for rel in ["02_tokenization/scripts/extract_vocab.py",
                "02_tokenization/scripts/clean_batch.py",
                "02_tokenization/scripts/tokenize_batch.py",
                "07_training/scripts/train.py",
                "09_finetuning/scripts/finetune.py",
                "08_evaluation/scripts/evaluate.py",
                "10_deployment/scripts/deploy.py"]:
        if not (REPO / rel).exists():
            failures.append(f"missing entrypoint referenced by orchestrator: {rel}")

    if failures:
        print("AUTOMATION TESTS FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All automation tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
