"""Layer normalization.

Normalizes each token's vector independently to mean 0 / unit variance across its
features, then applies a learned per-feature scale (gamma) and shift (beta). Keeps the
numbers well-conditioned for the sublayer that follows. Shape is preserved.

    x : (..., dim)  ->  (..., dim)
"""
from __future__ import annotations

import numpy as np


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)   # learned scale
        self.beta = np.zeros(dim, dtype=np.float32)   # learned shift
        self.grads: dict[str, np.ndarray] = {}
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)           # biased variance (population)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * inv_std
        self._cache = {"x_hat": x_hat, "inv_std": inv_std}
        return (self.gamma * x_hat + self.beta).astype(np.float32)

    __call__ = forward

    def backward(self, dy: np.ndarray) -> np.ndarray:
        x_hat, inv_std = self._cache["x_hat"], self._cache["inv_std"]
        dy = np.asarray(dy, dtype=np.float32)
        axes = tuple(range(dy.ndim - 1))              # sum over everything but features
        self.grads = {
            "gamma": (dy * x_hat).sum(axis=axes),
            "beta": dy.sum(axis=axes),
        }
        N = self.dim
        dx_hat = dy * self.gamma
        dx = (inv_std / N) * (
            N * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return dx.astype(np.float32)

    def parameters(self) -> dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    @property
    def num_parameters(self) -> int:
        return int(self.gamma.size + self.beta.size)
