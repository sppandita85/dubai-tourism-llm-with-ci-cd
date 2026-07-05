"""A single transformer block: the repeatable unit of the model.

Wires the attention and MLP components with the transformer glue: per-sublayer LayerNorm
(pre-norm) and residual (skip) connections.

    x = x + attention(norm1(x))     # attention sublayer, residual add
    x = x + mlp(norm2(x))           # feed-forward sublayer, residual add

Forward + backward (for the training phase). Shape is preserved throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in ("04_attention/src", "05_mlp/src"):
    sys.path.insert(0, str(_REPO / _p))

from attention import CausalMultiHeadAttention  # noqa: E402
from mlp import FeedForward                       # noqa: E402
from layer_norm import LayerNorm                  # noqa: E402  (same src dir)


class TransformerBlock:
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        hidden_mult: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        self.emb_dim = emb_dim
        s_attn = seed
        s_ff = None if seed is None else seed + 1
        self.norm1 = LayerNorm(emb_dim)
        self.attn = CausalMultiHeadAttention(
            emb_dim, num_heads, dropout=dropout, bias=bias, seed=s_attn)
        self.norm2 = LayerNorm(emb_dim)
        self.ff = FeedForward(
            emb_dim, hidden_mult=hidden_mult, activation=activation,
            dropout=dropout, bias=bias, seed=s_ff)

    def forward(self, x: np.ndarray, training: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = x + self.attn(self.norm1(x), training=training, rng=rng)   # attention sublayer
        x = x + self.ff(self.norm2(x), training=training, rng=rng)     # feed-forward sublayer
        return x

    __call__ = forward

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Residual 2: x2 = x1 + ff(norm2(x1))  -> gradient splits between the two paths.
        dx = np.asarray(dout, dtype=np.float32)
        dff = self.ff.backward(dx)
        dx = dx + self.norm2.backward(dff)
        # Residual 1: x1 = x0 + attn(norm1(x0))
        dattn = self.attn.backward(dx)
        dx = dx + self.norm1.backward(dattn)
        return dx

    def parameters(self) -> dict[str, np.ndarray]:
        p = {}
        for name, comp in (("norm1", self.norm1), ("attn", self.attn),
                           ("norm2", self.norm2), ("ff", self.ff)):
            for k, v in comp.parameters().items():
                p[f"{name}.{k}"] = v
        return p

    def gradients(self) -> dict[str, np.ndarray]:
        g = {}
        for name, comp in (("norm1", self.norm1), ("attn", self.attn),
                           ("norm2", self.norm2), ("ff", self.ff)):
            for k, v in comp.grads.items():
                g[f"{name}.{k}"] = v
        return g

    @property
    def num_parameters(self) -> int:
        return (self.norm1.num_parameters + self.attn.num_parameters
                + self.norm2.num_parameters + self.ff.num_parameters)
