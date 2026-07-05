"""Input embedding for the from-scratch LLM: token embeddings + positional embeddings.

This is a MODEL COMPONENT, not a data-prep step. It maps a batch of integer token IDs
(from the tokenization phase) to continuous vectors the transformer can consume:

    input_embeddings[b, t] = token_embedding[token_id]  +  positional_embedding[t]

Implemented in NumPy. Forward produces the vectors; backward (used by the training phase)
accumulates gradients for the token/positional tables. Weight matrices start random and
are learned during training.

Shapes
    token_ids : (batch, seq_len)             int
    output    : (batch, seq_len, emb_dim)    float32
"""
from __future__ import annotations

import numpy as np


class InputEmbedding:
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        pos_type: str = "learned",
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if pos_type not in ("learned", "sinusoidal"):
            raise ValueError(f"pos_type must be 'learned' or 'sinusoidal', got {pos_type!r}")
        if pos_type == "sinusoidal" and emb_dim % 2 != 0:
            raise ValueError("sinusoidal positional embeddings require an even emb_dim")

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.pos_type = pos_type
        self.dropout = float(dropout)
        rng = np.random.default_rng(seed)

        # Token embedding table: one learnable row per vocab id. Small-normal init.
        self.tok_emb = rng.normal(0.0, 0.02, size=(vocab_size, emb_dim)).astype(np.float32)

        # Positional embedding table: one vector per position 0..context_length-1.
        if pos_type == "learned":
            self.pos_emb = rng.normal(
                0.0, 0.02, size=(context_length, emb_dim)
            ).astype(np.float32)
        else:  # sinusoidal — fixed, not learned
            self.pos_emb = self._sinusoidal_table(context_length, emb_dim)

        self.grads: dict[str, np.ndarray] = {}
        self._cache: dict = {}

    @staticmethod
    def _sinusoidal_table(length: int, dim: int) -> np.ndarray:
        pos = np.arange(length)[:, None]                       # (length, 1)
        i = np.arange(0, dim, 2)[None, :]                      # (1, dim/2)
        div = np.exp(i * -(np.log(10000.0) / dim))             # (1, dim/2)
        table = np.zeros((length, dim), dtype=np.float32)
        table[:, 0::2] = np.sin(pos * div)
        table[:, 1::2] = np.cos(pos * div)
        return table

    @property
    def num_parameters(self) -> int:
        n = self.tok_emb.size
        if self.pos_type == "learned":
            n += self.pos_emb.size
        return int(n)

    def parameters(self) -> dict[str, np.ndarray]:
        p = {"tok_emb": self.tok_emb}
        if self.pos_type == "learned":
            p["pos_emb"] = self.pos_emb
        return p

    def forward(self, token_ids: np.ndarray, training: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be 2-D (batch, seq_len), got {token_ids.shape}")
        _, seq_len = token_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len {seq_len} exceeds context_length {self.context_length}")
        if token_ids.max(initial=0) >= self.vocab_size or token_ids.min(initial=0) < 0:
            raise ValueError("token id out of range for vocab_size")

        tok = self.tok_emb[token_ids]              # (batch, seq_len, emb_dim)
        pos = self.pos_emb[:seq_len][None, :, :]   # (1, seq_len, emb_dim), broadcasts
        out = tok + pos

        mask = None
        if training and self.dropout > 0.0:
            rng = rng or np.random.default_rng()
            keep = 1.0 - self.dropout
            mask = (rng.random(out.shape) < keep) / keep   # inverted dropout
            out = out * mask
        self._cache = {"token_ids": token_ids, "seq_len": seq_len, "mask": mask}
        return out.astype(np.float32)

    __call__ = forward

    def backward(self, dout: np.ndarray) -> None:
        """Accumulate gradients for the embedding tables. Input layer -> no dx returned."""
        c = self._cache
        dout = np.asarray(dout, dtype=np.float32)
        if c.get("mask") is not None:
            dout = dout * c["mask"]

        d_tok = np.zeros_like(self.tok_emb)
        np.add.at(d_tok, c["token_ids"], dout)     # scatter-add into used rows
        self.grads = {"tok_emb": d_tok}

        if self.pos_type == "learned":
            d_pos = np.zeros_like(self.pos_emb)
            d_pos[:c["seq_len"]] = dout.sum(axis=0)  # sum over the batch
            self.grads["pos_emb"] = d_pos

    def save(self, path: str) -> None:
        np.savez(
            path,
            tok_emb=self.tok_emb,
            pos_emb=self.pos_emb,
            meta=np.array(
                [self.vocab_size, self.emb_dim, self.context_length,
                 1 if self.pos_type == "learned" else 0]
            ),
        )
