from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import nn

ADAMW_LR = 1e-3
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = ADAMW_LR,
        betas: tuple[float, float] = ADAMW_BETAS,
        eps: float = ADAMW_EPS,
        weight_decay: float = ADAMW_WEIGHT_DECAY,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                self._update_param(param, lr, beta1, beta2, eps, weight_decay)

        return loss

    def _update_param(
        self,
        param: nn.Parameter,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        state = self.state[param]
        if len(state) == 0:
            state["t"] = 0
            state["m"] = torch.zeros_like(param)
            state["v"] = torch.zeros_like(param)

        state["t"] += 1
        t = state["t"]
        grad = param.grad

        # Moving averages of the gradient (first moment) and of its square (second moment)
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

        # Weight decay shrinks the parameter directly, separately from the gradient-based update
        param -= lr * weight_decay * param

        # Both averages start at zero, so early steps are scaled up to correct that bias
        lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        param -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
