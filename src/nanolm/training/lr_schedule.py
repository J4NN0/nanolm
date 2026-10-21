from __future__ import annotations

import math


def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # Warm-up: grow linearly from 0 to the max learning rate
    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate

    # Cosine annealing: decrease smoothly from the max to the min learning rate
    if t <= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)

    # Post-annealing: stay at the min learning rate
    return min_learning_rate
