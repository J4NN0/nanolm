from __future__ import annotations

import os

import torch

from nanolm.generation import generate
from nanolm.tokenizer import Tokenizer
from nanolm.training.__main__ import default_device
from nanolm.training.__main__ import main as run_training
from nanolm.transformer import TransformerLM

# Dataset
DATASET_NAME = "TinyStoriesV2-GPT4"
TRAIN_TEXT_PATH = f"data/{DATASET_NAME}-train.txt"
VALID_TEXT_PATH = f"data/{DATASET_NAME}-valid.txt"

# Tokenizer
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

# Model
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0

# Training: tokens processed = batch size × steps × context length (32 × 5,000 × 256 ≈ 41M)
BATCH_SIZE = 32
MAX_ITERS = 5000
MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 200

# Sample generated at the end
SAMPLE_PROMPT = "Once upon a time"
SAMPLE_MAX_NEW_TOKENS = 256
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9

TOKENIZERS_DIR = "tokenizers"
CHECKPOINTS_DIR = "checkpoints"
TOKENIZER_PATH = os.path.join(TOKENIZERS_DIR, f"{DATASET_NAME}_{VOCAB_SIZE}.json")
TRAIN_TOKENS_PATH = f"data/{DATASET_NAME}-train_{VOCAB_SIZE}.bin"
VALID_TOKENS_PATH = f"data/{DATASET_NAME}-valid_{VOCAB_SIZE}.bin"
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, f"{DATASET_NAME}_{VOCAB_SIZE}.pt")


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
    argv = [
        "--train-path", TRAIN_TOKENS_PATH,
        "--valid-path", VALID_TOKENS_PATH,
        "--vocab-size", str(VOCAB_SIZE),
        "--context-length", str(CONTEXT_LENGTH),
        "--d-model", str(D_MODEL),
        "--num-layers", str(NUM_LAYERS),
        "--num-heads", str(NUM_HEADS),
        "--d-ff", str(D_FF),
        "--rope-theta", str(ROPE_THETA),
        "--batch-size", str(BATCH_SIZE),
        "--max-iters", str(MAX_ITERS),
        "--max-lr", str(MAX_LR),
        "--min-lr", str(MIN_LR),
        "--warmup-iters", str(WARMUP_ITERS),
        "--checkpoint-path", CHECKPOINT_PATH,
    ]
    # An existing checkpoint is continued: an interrupted run resumes, a finished one trains no further
    if os.path.exists(CHECKPOINT_PATH):
        argv.append("--resume")

    run_training(argv)


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
