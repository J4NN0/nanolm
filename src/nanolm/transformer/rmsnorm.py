from __future__ import annotations

import torch
from torch import nn

RMSNORM_EPS = 1e-5


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = RMSNORM_EPS,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)
