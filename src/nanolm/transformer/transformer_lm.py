from __future__ import annotations

import torch
from torch import nn
from nanolm.transformer.embedding import Embedding
from nanolm.transformer.linear import Linear
from nanolm.transformer.rmsnorm import RMSNorm
from nanolm.transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.context_length: int = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            self.layers.append(block)

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"Sequence length ({seq_len}) exceeds context_length ({self.context_length})")

        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)

        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.lm_head(self.ln_final(x))
