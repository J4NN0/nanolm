from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta: float = theta
        self.d_k: int = d_k
        self.max_seq_len: int = max_seq_len

        angles = self._init_angles(device)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def _init_angles(self, device: torch.device | None) -> torch.Tensor:
        # One rotation frequency per pair of dimensions: theta^(-2k / d_k) for k = 0, ..., d_k / 2 - 1
        pair_indices = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
        frequencies = 1.0 / (self.theta ** (pair_indices / self.d_k))

        # Angle for every (position, pair): position * frequency, shape (max_seq_len, d_k / 2)
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        return torch.outer(positions, frequencies)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated pairs back into their original positions: (e0, o0, e1, o1, ...)
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
