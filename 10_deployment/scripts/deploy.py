#!/usr/bin/env python
"""Deploy a trained checkpoint to Ollama.

Exports checkpoints/<version>/model.npz to a GGUF, writes a Modelfile, and runs
`ollama create <model_name>:<version>`. Optionally pushes to the ollama.com registry.

    python 10_deployment/scripts/deploy.py --checkpoint checkpoints/v0.2.0/model.npz --version v0.2.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent      # 10_deployment/
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from export_gguf import export_checkpoint  # noqa: E402


def run(cmd: list[str]) -> int:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to model.npz (relative to repo)")
    ap.add_argument("--version", required=True, help="tag, e.g. v0.2.0")
    ap.add_argument("--name", default=None, help="override Ollama model name")
    ap.add_argument("--push", action="store_true", help="also push to the ollama.com registry")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "deploy.yaml").read_text())
    name = args.name or cfg["model_name"]
    ckpt = REPO / args.checkpoint
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    # 1. Export GGUF next to the checkpoint.
    out_dir = ckpt.parent
    gguf_path = out_dir / "model.gguf"
    print(f"Exporting {args.checkpoint} -> {gguf_path.relative_to(REPO)}")
    model_cfg = export_checkpoint(ckpt, gguf_path)
    print(f"  gpt2 GGUF: {gguf_path.stat().st_size/1e6:.1f} MB "
          f"(emb_dim={model_cfg['emb_dim']}, layers={model_cfg['num_layers']}, "
          f"vocab={model_cfg['vocab_size']})")

    # 2. Write the Modelfile.
    p = cfg["parameters"]
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(
        f"FROM ./model.gguf\n"
        f"PARAMETER temperature {p['temperature']}\n"
        f"PARAMETER top_k {p['top_k']}\n"
        f"PARAMETER num_ctx {p['num_ctx']}\n"
    )
    print(f"  wrote {modelfile.relative_to(REPO)}")

    # 3. ollama create.
    tag = f"{name}:{args.version}"
    print(f"Creating Ollama model {tag}")
    if run(["ollama", "create", tag, "-f", str(modelfile)]) != 0:
        print("ollama create failed", file=sys.stderr)
        return 1
    print(f"  created — run it with:  ollama run {tag}")

    # 4. Optional push to the registry.
    pushed_as = None
    if args.push or cfg.get("push_to_registry"):
        ns = cfg.get("registry_namespace", "").strip()
        if not ns:
            print("push requested but registry_namespace is empty in deploy.yaml — skipping",
                  file=sys.stderr)
        else:
            remote = f"{ns}/{name}:{args.version}"
            run(["ollama", "cp", tag, remote])
            if run(["ollama", "push", remote]) == 0:
                pushed_as = remote
                print(f"  pushed to registry as {remote}")

    # 5. Log the deployment.
    rec = {
        "record_type": "deployment",
        "deployed_at": date.today().isoformat(),
        "checkpoint": args.checkpoint,
        "version": args.version,
        "ollama_model": tag,
        "gguf": str(gguf_path.relative_to(REPO)),
        "pushed_to_registry": pushed_as,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest = REPO / "01_training_input_data" / "manifests" / "training_runs.jsonl"
    with open(manifest, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"Logged deployment of {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
