from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (..., vocab_size), targets: (...) with the index of the correct next token
    # Subtracting the max leaves the loss unchanged but keeps exp() from overflowing
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits

    # -log softmax(o)[target] = log(sum_j exp(o_j)) - o_target
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_logits = torch.gather(shifted_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = log_sum_exp - target_logits

    return torch.mean(losses)
