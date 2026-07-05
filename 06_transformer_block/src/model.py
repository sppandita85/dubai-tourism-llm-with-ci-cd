"""The assembled model: a complete forward AND backward pass.

    token_ids (batch, seq)
        -> input embeddings            (batch, seq, emb_dim)     [embedding phase]
        -> N transformer blocks        (batch, seq, emb_dim)     [this phase]
        -> final LayerNorm             (batch, seq, emb_dim)
        -> output head (emb_dim -> V)  (batch, seq, vocab_size)  = next-token logits

forward() returns logits; loss_and_backward() runs the whole backward pass and fills every
component's .grads. parameters()/gradients() expose flat dicts for the optimizer. Weights
are random until the training phase learns them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "03_embeddings" / "src"))

from input_embedding import InputEmbedding      # noqa: E402
from transformer_block import TransformerBlock   # noqa: E402
from layer_norm import LayerNorm                 # noqa: E402


def _softmax_lastdim(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class GPTModel:
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        num_heads: int,
        num_layers: int,
        hidden_mult: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        pos_type: str = "learned",
        seed: int | None = None,
    ) -> None:
        self.config = dict(
            vocab_size=vocab_size, emb_dim=emb_dim, context_length=context_length,
            num_heads=num_heads, num_layers=num_layers, hidden_mult=hidden_mult,
            activation=activation, dropout=dropout, bias=bias, pos_type=pos_type,
        )
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.num_layers = num_layers

        self.emb = InputEmbedding(vocab_size, emb_dim, context_length,
                                  pos_type=pos_type, dropout=dropout, seed=seed)
        self.blocks = [
            TransformerBlock(emb_dim, num_heads, hidden_mult=hidden_mult,
                             activation=activation, dropout=dropout, bias=bias,
                             seed=None if seed is None else seed + 100 * (i + 1))
            for i in range(num_layers)
        ]
        self.final_norm = LayerNorm(emb_dim)
        rng = np.random.default_rng(None if seed is None else seed + 9999)
        self.out_head = rng.normal(0.0, 0.02, (emb_dim, vocab_size)).astype(np.float32)

        self.grads: dict[str, np.ndarray] = {}
        self._cache: dict = {}

    def forward(self, token_ids: np.ndarray, training: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
        x = self.emb(token_ids, training=training, rng=rng)
        for block in self.blocks:
            x = block(x, training=training, rng=rng)
        xf = self.final_norm(x)
        logits = xf @ self.out_head
        self._cache = {"xf": xf}
        return logits.astype(np.float32)

    __call__ = forward

    def loss_and_backward(self, token_ids: np.ndarray, targets: np.ndarray,
                          rng: np.random.Generator | None = None) -> float:
        """Forward (training mode), compute mean next-token loss, run full backward."""
        logits = self.forward(token_ids, training=True, rng=rng)
        loss, dlogits = cross_entropy_with_grad(logits, targets)
        self.backward(dlogits)
        return loss

    def backward(self, dlogits: np.ndarray) -> None:
        xf = self._cache["xf"]
        d_out_head = np.einsum("bti,btv->iv", xf, dlogits)
        dxf = dlogits @ self.out_head.T
        dx = self.final_norm.backward(dxf)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.emb.backward(dx)
        self.grads = {"out_head": d_out_head}

    # ---- parameter / gradient plumbing for the optimizer ----
    def parameters(self) -> dict[str, np.ndarray]:
        p = {f"emb.{k}": v for k, v in self.emb.parameters().items()}
        for i, block in enumerate(self.blocks):
            for k, v in block.parameters().items():
                p[f"block{i}.{k}"] = v
        for k, v in self.final_norm.parameters().items():
            p[f"final_norm.{k}"] = v
        p["out_head"] = self.out_head
        return p

    def gradients(self) -> dict[str, np.ndarray]:
        g = {f"emb.{k}": v for k, v in self.emb.grads.items()}
        for i, block in enumerate(self.blocks):
            for k, v in block.gradients().items():
                g[f"block{i}.{k}"] = v
        for k, v in self.final_norm.grads.items():
            g[f"final_norm.{k}"] = v
        g["out_head"] = self.grads["out_head"]
        return g

    @property
    def num_parameters(self) -> int:
        n = self.emb.num_parameters
        n += sum(b.num_parameters for b in self.blocks)
        n += self.final_norm.num_parameters
        n += int(self.out_head.size)
        return int(n)

    # ---- checkpointing ----
    def save(self, path: str) -> None:
        import json
        arrays = {k: v for k, v in self.parameters().items()}
        np.savez(path, __config__=np.frombuffer(
            json.dumps(self.config).encode("utf-8"), dtype=np.uint8), **arrays)

    @classmethod
    def load(cls, path: str) -> "GPTModel":
        import json
        data = np.load(path)
        cfg = json.loads(bytes(data["__config__"]).decode("utf-8"))
        model = cls(seed=0, **cfg)
        params = model.parameters()
        for k, arr in params.items():
            arr[...] = data[k]                       # in-place: keeps references
        return model


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean next-token cross-entropy. ~ln(vocab_size) for a random model."""
    loss, _ = cross_entropy_with_grad(logits, targets)
    return loss


def cross_entropy_with_grad(logits: np.ndarray, targets: np.ndarray):
    """Returns (mean_loss, dlogits) where dlogits has the same shape as logits."""
    B, T, V = logits.shape
    flat = logits.reshape(-1, V)
    tgt = np.asarray(targets).reshape(-1)
    probs = _softmax_lastdim(flat)
    n = flat.shape[0]
    picked = probs[np.arange(n), tgt]
    loss = float(-np.log(np.clip(picked, 1e-12, None)).mean())
    dflat = probs
    dflat[np.arange(n), tgt] -= 1.0
    dflat /= n
    return loss, dflat.reshape(B, T, V).astype(np.float32)
