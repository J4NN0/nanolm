from __future__ import annotations

import torch

from nanolm.transformer import TransformerLM, softmax


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return softmax(logits / temperature, dim=-1)


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep the most likely tokens until their total reaches top_p: a token is kept
    # if the tokens ranked above it add up to less than top_p, so the first one is always kept
    keep = cumulative_probs - sorted_probs < top_p
    sorted_probs = sorted_probs * keep

    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(-1, sorted_ids, sorted_probs)
    return filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
) -> list[int]:
    if len(prompt_ids) == 0:
        raise ValueError("Prompt must contain at least one token")

    model.eval()
    device = next(model.parameters()).device
    token_ids = list(prompt_ids)

    new_token_ids: list[int] = []
    for _ in range(max_new_tokens):
        # The model accepts at most context_length tokens, so only the most recent ones are kept
        context = token_ids[-model.context_length :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)

        # Only the logits at the last position predict the next token: (1, seq_len, vocab_size) -> (vocab_size,)
        logits = model(inputs)[0, -1]

        probs = apply_temperature(logits, temperature)
        probs = top_p_filter(probs, top_p)
        next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        token_ids.append(next_token_id)
        new_token_ids.append(next_token_id)
        if next_token_id == end_token_id:
            break

    return new_token_ids
