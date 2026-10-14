from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Subtracting the max leaves the result unchanged but keeps exp() from overflowing
    max_values = torch.max(x, dim=dim, keepdim=True).values
    exp_values = torch.exp(x - max_values)
    return exp_values / torch.sum(exp_values, dim=dim, keepdim=True)
