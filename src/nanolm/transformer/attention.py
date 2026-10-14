from __future__ import annotations

import math
import torch
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
