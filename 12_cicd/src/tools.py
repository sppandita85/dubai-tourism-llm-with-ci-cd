"""CI agent tool implementations — plain Python (no framework).

Each returns a short human/LLM-readable string. graph.py wraps these as LangGraph tools;
run_ci_agent.py --no-llm calls them directly. Kept framework-free so the deterministic path
and the tests don't need LangGraph.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd)}"
    except Exception as e:  # noqa: BLE001
        return 1, f"error running {' '.join(cmd)}: {e}"


def git_sync(repo_url: str, branch: str, workspace: str) -> str:
    """Clone (shallow) or update the repo/branch into workspace; return the commit tested."""
    ws = Path(workspace)
    if (ws / ".git").exists():
        _run(["git", "-C", str(ws), "fetch", "--depth", "1", "origin", branch], timeout=180)
        rc, out = _run(["git", "-C", str(ws), "reset", "--hard", f"origin/{branch}"], timeout=60)
    else:
        ws.parent.mkdir(parents=True, exist_ok=True)
        rc, out = _run(["git", "clone", "--depth", "1", "-b", branch, repo_url, str(ws)],
                       timeout=300)
    if rc != 0:
        return f"git_sync FAILED (rc={rc}): {out.strip()[:400]}"
    _, h = _run(["git", "-C", str(ws), "log", "-1", "--pretty=%h %s"], timeout=30)
    return f"git_sync OK — checked out {branch} @ {h.strip()}  (workspace: {workspace})"


def run_tests(target_dir: str, runner: str) -> str:
    """Run every phase test suite against target_dir via run_all_tests.sh; summarize."""
    rc, out = _run(["bash", runner, target_dir], timeout=1200)
    lines = out.splitlines()
    result = next((l for l in lines if l.startswith("=== result:")), "")
    per = [l.strip() for l in lines if l.strip().startswith(("PASS", "FAIL"))]
    failed = next((l for l in lines if l.startswith("failed:")), "")
    verdict = "TESTS PASSED" if rc == 0 else "TESTS FAILED"
    body = "\n".join(per)
    tail = f"\n{failed}" if failed else ""
    return f"{verdict} (exit {rc})\n{result.replace('=== ','').replace(' ===','')}\n{body}{tail}"


def check_ollama_model(model: str, prompt: str, host: str) -> str:
    """Generate from a deployed Ollama model to prove it responds. Returns the text."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
                          "options": {"num_predict": 40}}).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        text = data.get("response", "").strip().replace("\n", " ")
        if not text:
            return f"MODEL CHECK FAILED — {model} returned empty output"
        return f"MODEL OK — {model} responded: {text[:240]}"
    except Exception as e:  # noqa: BLE001
        return f"MODEL CHECK FAILED — could not reach {model} at {host}: {e}"


def read_manifest(target_dir: str, n: int = 4) -> str:
    """Return the last n records from the pipeline's training_runs manifest."""
    mf = Path(target_dir) / "01_training_input_data" / "manifests" / "training_runs.jsonl"
    if not mf.exists():
        return f"no manifest at {mf}"
    rows = [l for l in mf.read_text().splitlines() if l.strip()][-n:]
    out = []
    for r in rows:
        try:
            d = json.loads(r)
            kind = d.get("record_type", d.get("strategy", "run"))
            out.append(f"  {kind}: {d.get('model_version') or d.get('checkpoint','')} "
                       f"{d.get('eval', '')}")
        except Exception:  # noqa: BLE001
            out.append("  " + r[:120])
    return "recent runs:\n" + "\n".join(out)
