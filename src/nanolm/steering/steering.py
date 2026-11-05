from __future__ import annotations

import torch
from torch import nn

from nanolm.transformer import TransformerLM


class Steering:

    def __init__(
        self,
        model: TransformerLM,
        layer: int,
        vector: torch.Tensor,
        strength: float,
        num_positions: int | None = None,
    ) -> None:
        if not 1 <= layer <= len(model.layers):
            raise ValueError(f"layer must be in [1, {len(model.layers)}], got {layer}")

        self.block: nn.Module = model.layers[layer - 1]
        self.delta: torch.Tensor = strength * vector
        # None adds the vector at every position; a number only at the first ones (e.g. the prompt).
        # Positions count from the start of the model input, so this is exact only while the text fits in the context
        self.num_positions: int | None = num_positions
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    # Used as `with Steering(...):` so the vector is only added inside the block, and always removed after
    def __enter__(self) -> Steering:
        self.handle = self.block.register_forward_hook(self._add_delta)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _add_delta(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        # Replaces the block output: (batch, seq_len, d_model) + (d_model,) adds the same vector at every position
        if self.num_positions is None:
            return output + self.delta

        steered = output.clone()
        steered[:, : self.num_positions] += self.delta
        return steered


@torch.no_grad()
def residual_stream(model: TransformerLM, token_ids: list[int], layer: int) -> torch.Tensor:
    if not 1 <= layer <= len(model.layers):
        raise ValueError(f"layer must be in [1, {len(model.layers)}], got {layer}")
    if len(token_ids) == 0:
        raise ValueError("Text must contain at least one token")

    captured: list[torch.Tensor] = []

    def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output[0])

    model.eval()
    device = next(model.parameters()).device
    handle = model.layers[layer - 1].register_forward_hook(capture)
    try:
        model(torch.tensor([token_ids], dtype=torch.long, device=device))
    finally:
        handle.remove()

    # Output of block `layer` (counted from 1) for every token: (seq_len, d_model)
    return captured[0]


def steering_vector(
    model: TransformerLM,
    toward_ids: list[int],
    away_ids: list[int] | None,
    layer: int,
) -> torch.Tensor:
    # The two texts should differ only in the concept (e.g. "dragon" and "girl"): what they share cancels out,
    # and the difference at the last token keeps what separates the concepts
    toward = residual_stream(model, toward_ids, layer)[-1]
    if away_ids is None:
        return toward

    away = residual_stream(model, away_ids, layer)[-1]
    return toward - away
