"""RCA tools — evidence-gatherers with built-in detection.

Each returns a compact, structured findings string (with OK/PROBLEM flags) so a weak LLM can
synthesize a root cause without doing the detection itself. All read-only except run_tests.
Paths are resolved against the repo root (parents[2] of this file).
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _get(url: str, timeout: int = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def check_environment(required_models: list[str], ollama_host: str) -> str:
    """Check Ollama, required models, per-phase venvs, and the tokenizer vocab."""
    out = ["ENVIRONMENT:"]
    # Ollama + models
    try:
        _get(f"{ollama_host}/api/version")
        tags = _get(f"{ollama_host}/api/tags")
        have = {m["name"] for m in tags.get("models", [])}
        out.append("  OK    ollama reachable")
        for m in required_models:
            hit = any(h == m or h.split(":")[0] == m.split(":")[0] for h in have)
            out.append(f"  {'OK   ' if hit else 'PROBLEM'} model {m}"
                       f"{'' if hit else ' MISSING'}")
    except Exception as e:  # noqa: BLE001
        out.append(f"  PROBLEM ollama unreachable at {ollama_host}: {e}")
    # per-phase venvs (01_training_input_data has none by design; 11_automation uses system python3)
    no_venv_by_design = {"01_training_input_data", "11_automation"}
    missing_venv = []
    for p in sorted(pp.name for pp in REPO.glob("[0-9][0-9]_*") if pp.is_dir()):
        if p in no_venv_by_design:
            continue
        needs = (REPO / p / "src").exists() or (REPO / p / "scripts").exists()
        has = (REPO / p / ".venv" / "bin" / "python").exists()
        if needs and not has:
            missing_venv.append(p)
    out.append(f"  {'OK   ' if not missing_venv else 'PROBLEM'} per-phase venvs"
               f"{'' if not missing_venv else ' MISSING: ' + ', '.join(missing_venv)}")
    # tokenizer vocab
    vocab = REPO / "02_tokenization" / "vocab" / "nomic-embed-text" / "tokenizer.json"
    out.append(f"  {'OK   ' if vocab.exists() else 'PROBLEM'} tokenizer vocab"
               f"{'' if vocab.exists() else ' MISSING (run extract_vocab.py)'}")
    return "\n".join(out)


def scan_logs(log_globs: list[str], max_files: int = 6) -> str:
    """Collect newest logs and extract error/failure/traceback lines with context."""
    files = []
    for g in log_globs:
        files += list(REPO.glob(g))
    # never scan our own RCA reports (avoids a self-referential feedback loop)
    files = [f for f in files if "13_troubleshooting/logs" not in str(f)]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    if not files:
        return "LOGS: no recent log files found (pipeline/CI may not have run here)."
    def is_issue(line: str) -> bool:
        low = line.lower()
        # ignore success summaries and prompt-echo/instruction lines
        if "0 failed" in low or "passed" in low or "verdict:" in low or low.strip().startswith("or"):
            return False
        if "traceback" in low or "exception" in low or "error" in low:
            return True
        # a real test-runner failure line is "FAIL <suite>"; or an actual FAIL verdict
        return low.lstrip().startswith("fail") or "problem" in low

    out = ["LOGS (newest first):"]
    for f in files:
        try:
            lines = f.read_text(errors="replace").splitlines()
        except Exception:  # noqa: BLE001
            continue
        hits = [l.strip() for l in lines if is_issue(l)]
        rel = f.relative_to(REPO)
        if hits:
            out.append(f"  {rel}: {len(hits)} issue line(s):")
            out += [f"      {h[:160]}" for h in hits[:4]]
        else:
            out.append(f"  {rel}: clean (no error/fail lines)")
    return "\n".join(out)


def analyze_manifests(manifest: str) -> str:
    """Parse training_runs.jsonl; report latest metrics and flag anomalies (overfitting, etc.)."""
    mf = REPO / manifest
    if not mf.exists():
        return f"MANIFEST: not found at {manifest}"
    records = []
    for line in mf.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    if not records:
        return "MANIFEST: empty."

    trains = [r for r in records if r.get("strategy") in ("from_scratch", "continued")]
    evals = [r for r in records if r.get("record_type") == "evaluation"]
    deploys = [r for r in records if r.get("record_type") == "deployment"]
    out = ["MANIFEST ANALYSIS:"]
    if trains:
        t = trains[-1]
        ev = t.get("eval", {})
        tl = ev.get("final_train_loss"); vl = ev.get("final_val_loss")
        out.append(f"  latest training: {t.get('model_version')} "
                   f"train_loss={tl} val_loss={vl} strategy={t.get('strategy')}")
        if tl is not None and vl is not None and vl > tl + 2.0:
            out.append(f"  PROBLEM overfitting: val_loss ({vl}) >> train_loss ({tl}) "
                       f"— model memorized the training data")
    if evals:
        e = evals[-1].get("eval", {})
        ppl = e.get("perplexity")
        out.append(f"  latest evaluation: perplexity={ppl} holdout_loss={e.get('holdout_loss')}")
        if ppl is not None and ppl > 1000:
            out.append(f"  PROBLEM very high perplexity ({ppl}) — poor generalization "
                       f"(tiny single-document corpus)")
    if trains and not deploys:
        out.append("  PROBLEM trained but no deployment record found")
    if not any("PROBLEM" in l for l in out):
        out.append("  OK    no manifest anomalies detected")
    return "\n".join(out)


def run_tests(test_runner: str, timeout: int = 1200) -> str:
    """Run the phase test suites via 12_cicd/run_all_tests.sh; return failures + tracebacks."""
    runner = REPO / test_runner
    if not runner.exists():
        return f"RUN_TESTS: runner not found at {test_runner}"
    try:
        p = subprocess.run(["bash", str(runner)], cwd=str(REPO),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"RUN_TESTS: timeout after {timeout}s"
    out = p.stdout + p.stderr
    result = next((l for l in out.splitlines() if l.startswith("=== result:")), "")
    fails = [l for l in out.splitlines() if l.strip().startswith("FAIL") or l.startswith("failed:")]
    status = "ALL TESTS PASSED" if p.returncode == 0 else "TEST FAILURES"
    body = "\n".join(f"  {l.strip()}" for l in fails) if fails else "  (none)"
    return f"RUN_TESTS: {status}\n  {result}\n{body}"


def read_source(path_or_glob: str, pattern: str | None = None, max_lines: int = 60) -> str:
    """Read a bounded slice of a repo file (optionally only lines matching a grep pattern)."""
    matches = sorted(REPO.glob(path_or_glob)) if any(c in path_or_glob for c in "*?[") \
        else [REPO / path_or_glob]
    matches = [m for m in matches if m.is_file()][:3]
    if not matches:
        return f"READ_SOURCE: no file matched {path_or_glob}"
    out = []
    for m in matches:
        lines = m.read_text(errors="replace").splitlines()
        if pattern:
            picked = [f"{i+1}: {l}" for i, l in enumerate(lines) if pattern in l][:max_lines]
        else:
            picked = [f"{i+1}: {l}" for i, l in enumerate(lines[:max_lines])]
        out.append(f"--- {m.relative_to(REPO)} ---\n" + "\n".join(picked))
    return "\n".join(out)
