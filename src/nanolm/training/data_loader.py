from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    x: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Each snippet needs context_length inputs plus one more token for the last target
    if len(x) <= context_length:
        raise ValueError(f"Dataset has {len(x)} tokens, needs more than context_length ({context_length})")

    # Random start of every snippet: the last valid start is len(x) - context_length - 1
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    inputs = np.empty((batch_size, context_length), dtype=np.int64)
    targets = np.empty((batch_size, context_length), dtype=np.int64)
    for row, start in enumerate(starts):
        inputs[row] = x[start : start + context_length]
        # Targets are the inputs shifted by one: the token that follows each input token
        targets[row] = x[start + 1 : start + context_length + 1]

    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
