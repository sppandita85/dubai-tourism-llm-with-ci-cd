"""Position-wise feed-forward network (MLP) for the from-scratch LLM.

In a transformer block this follows attention. Attention mixes information *across*
positions; the MLP then transforms each position independently through an expand ->
nonlinearity -> project bottleneck:

    hidden = gelu(x @ W1 + b1)      # (…, emb_dim) -> (…, hidden = 4*emb_dim)
    out    = hidden @ W2 + b2       # (…, hidden)  -> (…, emb_dim)

Shape preserved: (batch, seq_len, emb_dim) -> (batch, seq_len, emb_dim). NumPy forward +
backward (for the training phase); weights start random and are learned.
"""
from __future__ import annotations

import numpy as np

_GELU_K = np.sqrt(2.0 / np.pi)
_GELU_C = 0.044715


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU with the tanh approximation used by GPT-2."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_K * (x + _GELU_C * x ** 3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """d gelu / d x for the tanh approximation."""
    u = _GELU_K * (x + _GELU_C * x ** 3)
    t = np.tanh(u)
    du = _GELU_K * (1.0 + 3.0 * _GELU_C * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


ACTIVATIONS = {"gelu": (gelu, gelu_grad), "relu": (relu, relu_grad)}


class FeedForward:
    def __init__(
        self,
        emb_dim: int,
        hidden_mult: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}, got {activation!r}")
        self.emb_dim = emb_dim
        self.hidden = emb_dim * hidden_mult
        self.activation_name = activation
        self.activation, self.activation_grad = ACTIVATIONS[activation]
        self.dropout = float(dropout)
        self.bias = bias
        rng = np.random.default_rng(seed)

        self.W1 = (rng.normal(0.0, 1.0, (emb_dim, self.hidden))
                   * np.sqrt(2.0 / emb_dim)).astype(np.float32)
        self.W2 = (rng.normal(0.0, 1.0, (self.hidden, emb_dim))
                   * np.sqrt(2.0 / self.hidden)).astype(np.float32)
        self.b1 = np.zeros(self.hidden, dtype=np.float32) if bias else None
        self.b2 = np.zeros(emb_dim, dtype=np.float32) if bias else None

        self.grads: dict[str, np.ndarray] = {}
        self._cache: dict = {}

    def forward(self, x: np.ndarray, training: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.emb_dim:
            raise ValueError(f"last dim must be emb_dim={self.emb_dim}, got {x.shape[-1]}")

        h_pre = x @ self.W1
        if self.b1 is not None:
            h_pre = h_pre + self.b1
        a = self.activation(h_pre)
        out = a @ self.W2
        if self.b2 is not None:
            out = out + self.b2

        mask = None
        if training and self.dropout > 0.0:
            rng = rng or np.random.default_rng()
            keep = 1.0 - self.dropout
            mask = (rng.random(out.shape) < keep) / keep
            out = out * mask
        self._cache = {"x": x, "h_pre": h_pre, "a": a, "mask": mask}
        return out.astype(np.float32)

    __call__ = forward

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        dout = np.asarray(dout, dtype=np.float32)
        if c["mask"] is not None:
            dout = dout * c["mask"]

        a, x, h_pre = c["a"], c["x"], c["h_pre"]
        dW2 = np.einsum("bth,bte->he", a, dout)
        da = dout @ self.W2.T
        dh = da * self.activation_grad(h_pre)
        dW1 = np.einsum("bte,bth->eh", x, dh)
        dx = dh @ self.W1.T

        self.grads = {"W1": dW1, "W2": dW2}
        if self.bias:
            self.grads["b1"] = dh.sum(axis=(0, 1))
            self.grads["b2"] = dout.sum(axis=(0, 1))
        return dx.astype(np.float32)

    def parameters(self) -> dict[str, np.ndarray]:
        p = {"W1": self.W1, "W2": self.W2}
        if self.bias:
            p["b1"], p["b2"] = self.b1, self.b2
        return p

    @property
    def num_parameters(self) -> int:
        n = self.emb_dim * self.hidden + self.hidden * self.emb_dim
        if self.bias:
            n += self.hidden + self.emb_dim
        return int(n)
