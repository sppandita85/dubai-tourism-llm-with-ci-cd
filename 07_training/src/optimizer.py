"""AdamW optimizer (NumPy).

Adam with decoupled weight decay. Updates parameter arrays in place so the model keeps
its references. Weight decay is applied only to matrices (ndim >= 2), not to biases or
LayerNorm gains — the standard convention.
"""
from __future__ import annotations

import numpy as np


class AdamW:
    def __init__(self, lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        bias_c1 = 1.0 - self.b1 ** self.t
        bias_c2 = 1.0 - self.b2 ** self.t
        for name, p in params.items():
            g = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            m, v = self.m[name], self.v[name]
            m[...] = self.b1 * m + (1.0 - self.b1) * g
            v[...] = self.b2 * v + (1.0 - self.b2) * (g * g)
            m_hat = m / bias_c1
            v_hat = v / bias_c2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.wd > 0.0 and p.ndim >= 2:            # decoupled weight decay
                update = update + self.wd * p
            p[...] = p - self.lr * update                # in place: keeps references
