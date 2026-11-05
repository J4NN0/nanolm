from __future__ import annotations

import os

import torch

from nanolm.config import (
    CHECKPOINT_PATH,
    CONTEXT_LENGTH,
    D_FF,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    ROPE_THETA,
    SPECIAL_TOKENS,
    TOKENIZER_PATH,
    TRAIN_TEXT_PATH,
    TRAIN_TOKENS_PATH,
    VALID_TEXT_PATH,
    VALID_TOKENS_PATH,
    VOCAB_SIZE,
)
from nanolm.generation import generate
from nanolm.tokenizer import Tokenizer
from nanolm.training.__main__ import default_device
from nanolm.training.__main__ import main as run_training
from nanolm.transformer import TransformerLM

SAMPLE_PROMPT = "Once upon a time"
SAMPLE_MAX_NEW_TOKENS = 256
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9


def load_or_train_tokenizer() -> Tokenizer:
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH} ...")
        return Tokenizer.load(TOKENIZER_PATH)

    print(f"Training tokenizer on {TRAIN_TEXT_PATH} ...")
    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)
    tokenizer.train(input_path=TRAIN_TEXT_PATH, vocab_size=VOCAB_SIZE)
    tokenizer.save(TOKENIZER_PATH)

    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    return tokenizer


def encode_dataset(tokenizer: Tokenizer, text_path: str, tokens_path: str) -> None:
    if os.path.exists(tokens_path):
        print(f"Found token IDs in {tokens_path}")
        return

    print(f"Encoding {text_path} ...")
    # Written under a temporary name first, so an interrupted run never leaves a partial file behind
    temp_path = tokens_path + ".tmp"
    num_tokens = tokenizer.encode_file(text_path, temp_path)
    os.replace(temp_path, tokens_path)

    print(f"Saved {num_tokens:,} token IDs to {tokens_path}")


def train_model() -> None:
    # Every setting comes from nanolm.config, so no options are passed
    run_training([])


def generate_sample(tokenizer: Tokenizer) -> None:
    device = default_device()
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        device=device,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    end_token_id = tokenizer.token_ids[SPECIAL_TOKENS[0].encode("utf-8")]
    prompt_ids = tokenizer.encode(SAMPLE_PROMPT)
    new_token_ids = generate(model, prompt_ids, SAMPLE_MAX_NEW_TOKENS, SAMPLE_TEMPERATURE, SAMPLE_TOP_P, end_token_id)

    print("Sample:")
    print(SAMPLE_PROMPT + tokenizer.decode(new_token_ids))


def main() -> None:
    tokenizer = load_or_train_tokenizer()
    encode_dataset(tokenizer, TRAIN_TEXT_PATH, TRAIN_TOKENS_PATH)
    encode_dataset(tokenizer, VALID_TEXT_PATH, VALID_TOKENS_PATH)
    train_model()
    generate_sample(tokenizer)


if __name__ == "__main__":
    main()
