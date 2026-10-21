from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

GRADIENT_CLIPPING_EPS = 1e-6


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = GRADIENT_CLIPPING_EPS,
) -> None:
    grads: list[torch.Tensor] = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)

    if len(grads) == 0:
        return

    # The norm is taken over all gradients together, as if they were one long vector
    squared_sum = torch.zeros((), device=grads[0].device)
    for grad in grads:
        squared_sum += torch.sum(grad.to(torch.float32) ** 2)
    total_norm = torch.sqrt(squared_sum).item()

    if total_norm <= max_l2_norm:
        return

    # Scaling every gradient by the same factor keeps the update direction unchanged
    scale = max_l2_norm / (total_norm + eps)
    for grad in grads:
        grad.mul_(scale)
