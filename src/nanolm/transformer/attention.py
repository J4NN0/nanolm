from __future__ import annotations

import math
import torch
from torch import nn
from nanolm.transformer.linear import Linear
from nanolm.transformer.rope import RotaryPositionalEmbedding
from nanolm.transformer.softmax import softmax


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = queries.shape[-1]

    # (..., queries, d_k) @ (..., d_k, keys) -> (..., queries, keys)
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(d_k)

    # Positions where the mask is False get -inf, so softmax gives them zero weight
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)

    # (..., queries, keys) @ (..., keys, d_v) -> (..., queries, d_v)
    return attention_weights @ values


class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_k: int = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope: RotaryPositionalEmbedding | None = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # Add a heads dimension so the same rotation is applied in every head: (..., 1, seq_len)
            head_positions = token_positions.unsqueeze(-2)
            queries = self.rope(queries, head_positions)
            keys = self.rope(keys, head_positions)

        # Token i may attend to tokens 0, ..., i: lower-triangular (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attention_output = scaled_dot_product_attention(queries, keys, values, causal_mask)
        return self.output_proj(self._merge_heads(attention_output))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
        x = x.reshape(*x.shape[:-1], self.num_heads, self.d_k)
        return x.transpose(-3, -2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., num_heads, seq_len, d_k) -> (..., seq_len, num_heads, d_k) -> (..., seq_len, d_model)
        x = x.transpose(-3, -2)
        return x.reshape(*x.shape[:-2], self.d_model)
