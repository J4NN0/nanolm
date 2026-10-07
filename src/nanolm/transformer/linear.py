from __future__ import annotations

import math
import torch
from torch import nn

# Weights are drawn from a normal distribution truncated at this many standard deviations
TRUNCATION_STDS = 3


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-TRUNCATION_STDS * std, b=TRUNCATION_STDS * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
