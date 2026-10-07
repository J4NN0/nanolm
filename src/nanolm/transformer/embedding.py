from __future__ import annotations

import torch
from torch import nn
from nanolm.transformer.linear import TRUNCATION_STDS

EMBEDDING_INIT_STD = 1.0


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self) -> None:
        std = EMBEDDING_INIT_STD
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-TRUNCATION_STDS * std, b=TRUNCATION_STDS * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
