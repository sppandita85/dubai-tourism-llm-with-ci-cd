"""Shared helpers for the tokenization phase."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

# tokenization/ (parent of scripts/)
TOK_ROOT = Path(__file__).resolve().parent.parent
# repo root (parent of tokenization/)
REPO_ROOT = TOK_ROOT.parent
DATA_ROOT = REPO_ROOT / "01_training_input_data"
CONFIG_PATH = TOK_ROOT / "config" / "tokenizer.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def vocab_dir(cfg: dict) -> Path:
    return TOK_ROOT / "vocab" / cfg["vocab_dir"]


def resolve_model_blob(cfg: dict) -> tuple[Path, str]:
    """Return (path, digest) of the GGUF model blob for cfg['source_model'].

    Reads the Ollama manifest to map a model name/tag to the blob whose
    mediaType is the model image, then locates that blob on disk.
    """
    models_dir = Path(os.path.expanduser(cfg["ollama_models_dir"]))
    name, _, tag = cfg["source_model"].partition(":")
    tag = tag or "latest"
    # Ollama library models live under registry.ollama.ai/library/<name>/<tag>.
    manifest = (
        models_dir / "manifests" / "registry.ollama.ai" / "library" / name / tag
    )
    if not manifest.exists():
        raise FileNotFoundError(f"Ollama manifest not found: {manifest}")
    meta = json.loads(manifest.read_text())
    digest = None
    for layer in meta.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"]
            break
    if digest is None:
        raise ValueError(f"No model layer in manifest for {cfg['source_model']}")
    blob = models_dir / "blobs" / digest.replace(":", "-")
    if not blob.exists():
        raise FileNotFoundError(f"Model blob missing: {blob}")
    return blob, digest


def batch_dir(stage: str, batch: str) -> Path:
    """Path to a stage folder (raw/interim/processed) for a given YYYY-MM batch."""
    return DATA_ROOT / stage / batch
