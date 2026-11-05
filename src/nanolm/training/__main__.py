from __future__ import annotations

import argparse
import os
import time

import numpy as np
import numpy.typing as npt
import torch

from nanolm.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CONTEXT_LENGTH,
    D_FF,
    D_MODEL,
    MAX_ITERS,
    MAX_LR,
    MIN_LR,
    NUM_HEADS,
    NUM_LAYERS,
    ROPE_THETA,
    TRAIN_TOKENS_PATH,
    VALID_TOKENS_PATH,
    VOCAB_SIZE,
    WARMUP_ITERS,
)
from nanolm.tokenizer.tokenizer import TOKEN_ID_DTYPE
from nanolm.training import AdamW, clip_gradients, cross_entropy, get_batch, get_lr_cosine_schedule
from nanolm.training.checkpointing import load_checkpoint, save_checkpoint
from nanolm.transformer import TransformerLM

# AdamW and gradient clipping: the standard values, which almost every training run keeps
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

# Logging and checkpoints: how often to print and save, which does not change the trained model
LOG_INTERVAL = 10
EVAL_INTERVAL = 250
EVAL_ITERS = 20
CHECKPOINT_INTERVAL = 1000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model with the settings in nanolm.config")

    parser.add_argument(
        "--checkpoint-path",
        default=CHECKPOINT_PATH,
        help="Where the model is saved; training continues from it if it exists",
    )

    parser.add_argument("--device", default=None, help="e.g. cpu, cuda:0, mps (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    # argv is None when run from the command line, so the options are read from sys.argv
    return parser.parse_args(argv)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokens(path: str) -> npt.NDArray[np.integer]:
    # Memory-mapped: the file stays on disk and only the sampled snippets are read into memory
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=TOKEN_ID_DTYPE, mode="r")


@torch.no_grad()
def evaluate(model: TransformerLM, data: npt.NDArray[np.integer], device: str) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(EVAL_ITERS):
        inputs, targets = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH, device)
        total_loss += cross_entropy(model(inputs), targets).item()
    model.train()
    return total_loss / EVAL_ITERS


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or default_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = load_tokens(TRAIN_TOKENS_PATH)
    valid_data = load_tokens(VALID_TOKENS_PATH)
    print(f"Train tokens: {len(train_data):,}, validation tokens: {len(valid_data):,}, device: {device}")

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
    optimizer = AdamW(
        model.parameters(),
        lr=MAX_LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"Model parameters: {num_params:,}")

    start_iteration = 0
    # An existing checkpoint is continued: an interrupted run resumes, a finished one trains no further
    if os.path.exists(args.checkpoint_path):
        start_iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from {args.checkpoint_path} at iteration {start_iteration}")

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    for iteration in range(start_iteration, MAX_ITERS):
        lr = get_lr_cosine_schedule(iteration, MAX_LR, MIN_LR, WARMUP_ITERS, MAX_ITERS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, device)
        loss = cross_entropy(model(inputs), targets)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # Iterations are counted as completed steps, so a checkpoint saved at step n resumes at step n
        step = iteration + 1
        is_last_step = step == MAX_ITERS

        if step % LOG_INTERVAL == 0 or is_last_step:
            elapsed = time.time() - start_time
            print(f"step {step}/{MAX_ITERS} | train loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        if step % EVAL_INTERVAL == 0 or is_last_step:
            valid_loss = evaluate(model, valid_data, device)
            print(f"step {step}/{MAX_ITERS} | validation loss {valid_loss:.4f}")

        if step % CHECKPOINT_INTERVAL == 0 or is_last_step:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
