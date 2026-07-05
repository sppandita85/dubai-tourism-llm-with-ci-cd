"""Causal multi-head self-attention for the from-scratch LLM.

The layer where the model mixes information across positions. Each position produces a
query, key, and value; a position attends to itself and to *earlier* positions only (the
causal mask), which makes the model autoregressive.

    x : (batch, seq_len, emb_dim)   input
    y : (batch, seq_len, emb_dim)   context-mixed output

NumPy forward + backward (for the training phase). Every step — projections, scaled
dot-product, masking, softmax, head merge — is explicit. Weights are learned in training.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)      # stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class CausalMultiHeadAttention:
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout = float(dropout)
        self.bias = bias
        rng = np.random.default_rng(seed)

        def w():
            return rng.normal(0.0, 0.02, size=(emb_dim, emb_dim)).astype(np.float32)

        def b():
            return np.zeros(emb_dim, dtype=np.float32) if bias else None

        self.W_q, self.W_k, self.W_v, self.W_o = w(), w(), w(), w()
        self.b_q, self.b_k, self.b_v, self.b_o = b(), b(), b(), b()
        self.last_attn: np.ndarray | None = None   # (batch, heads, t, t) for inspection
        self.grads: dict[str, np.ndarray] = {}
        self._cache: dict = {}

    def _project(self, x, W, bvec):
        y = x @ W
        return y + bvec if bvec is not None else y

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, h, t, hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * hd)

    def forward(self, x: np.ndarray, training: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[2] != self.emb_dim:
            raise ValueError(f"expected (batch, seq_len, {self.emb_dim}), got {x.shape}")
        _, t, _ = x.shape

        q = self._split_heads(self._project(x, self.W_q, self.b_q))  # (b,h,t,hd)
        k = self._split_heads(self._project(x, self.W_k, self.b_k))
        v = self._split_heads(self._project(x, self.W_v, self.b_v))

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale          # (b,h,t,t)
        future = np.triu(np.ones((t, t), dtype=bool), k=1)      # position i can't see j>i
        scores = np.where(future, -np.inf, scores)
        attn = softmax(scores, axis=-1)

        mask = None
        attn_d = attn
        if training and self.dropout > 0.0:
            rng = rng or np.random.default_rng()
            keep = 1.0 - self.dropout
            mask = ((rng.random(attn.shape) < keep) / keep).astype(attn.dtype)
            attn_d = attn * mask
        self.last_attn = attn

        ctx = attn_d @ v                                        # (b,h,t,hd)
        ctx_merged = self._merge_heads(ctx)                     # (b,t,emb)
        out = self._project(ctx_merged, self.W_o, self.b_o)

        self._cache = {"x": x, "q": q, "k": k, "v": v, "attn": attn, "attn_d": attn_d,
                       "ctx_merged": ctx_merged, "future": future, "scale": scale,
                       "mask": mask}
        return out

    __call__ = forward

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        dout = np.asarray(dout)
        x, q, k, v = c["x"], c["q"], c["k"], c["v"]
        attn, attn_d, ctx_merged = c["attn"], c["attn_d"], c["ctx_merged"]
        scale, future = c["scale"], c["future"]

        # Output projection: out = ctx_merged @ W_o + b_o
        dW_o = np.einsum("bti,btj->ij", ctx_merged, dout)
        dctx_merged = dout @ self.W_o.T
        dctx = self._split_heads(dctx_merged)                  # (b,h,t,hd)

        # ctx = attn_d @ v
        dattn_d = dctx @ v.transpose(0, 1, 3, 2)               # (b,h,t,t)
        dv = attn_d.transpose(0, 1, 3, 2) @ dctx               # (b,h,t,hd)

        # dropout on attention weights
        dattn = dattn_d * c["mask"] if c["mask"] is not None else dattn_d

        # softmax backward (per row over the last axis)
        dscores = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        dscores = np.where(future, 0.0, dscores)

        # scores = (q @ k^T) * scale
        dq = (dscores @ k) * scale                             # (b,h,t,hd)
        dk = (dscores.transpose(0, 1, 3, 2) @ q) * scale       # (b,h,t,hd)

        dq_m = self._merge_heads(dq)                           # (b,t,emb)
        dk_m = self._merge_heads(dk)
        dv_m = self._merge_heads(dv)

        # q/k/v = x @ W_{q,k,v} (+ bias)
        dW_q = np.einsum("bti,btj->ij", x, dq_m)
        dW_k = np.einsum("bti,btj->ij", x, dk_m)
        dW_v = np.einsum("bti,btj->ij", x, dv_m)
        dx = dq_m @ self.W_q.T + dk_m @ self.W_k.T + dv_m @ self.W_v.T

        self.grads = {"W_q": dW_q, "W_k": dW_k, "W_v": dW_v, "W_o": dW_o}
        if self.bias:
            self.grads["b_q"] = dq_m.sum(axis=(0, 1))
            self.grads["b_k"] = dk_m.sum(axis=(0, 1))
            self.grads["b_v"] = dv_m.sum(axis=(0, 1))
            self.grads["b_o"] = dout.sum(axis=(0, 1))
        return dx

    def parameters(self) -> dict[str, np.ndarray]:
        p = {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}
        if self.bias:
            p.update({"b_q": self.b_q, "b_k": self.b_k, "b_v": self.b_v, "b_o": self.b_o})
        return p

    @property
    def num_parameters(self) -> int:
        n = 4 * self.emb_dim * self.emb_dim
        if self.bias:
            n += 4 * self.emb_dim
        return int(n)
