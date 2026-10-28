from __future__ import annotations

import argparse
import os
import time

import numpy as np
import numpy.typing as npt
import torch

from nanolm.training import AdamW, clip_gradients, cross_entropy, get_batch, get_lr_cosine_schedule
from nanolm.training.checkpointing import load_checkpoint, save_checkpoint
from nanolm.transformer import TransformerLM

DEFAULT_TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.npy"
DEFAULT_VALID_PATH = "data/TinyStoriesV2-GPT4-valid.npy"
DEFAULT_CHECKPOINT_PATH = "checkpoints/checkpoint.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model on token ID files")

    # Data: .npy files, or raw binary files of --data-dtype token IDs
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--data-dtype", default="uint16")

    # Model
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Optimizer and learning rate schedule; the cosine decay ends at --max-iters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging and checkpoints
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Continue from the checkpoint at --checkpoint-path")

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


def load_tokens(path: str, dtype: str) -> npt.NDArray[np.integer]:
    # Memory-mapped: the file stays on disk and only the sampled snippets are read into memory
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=dtype, mode="r")


@torch.no_grad()
def evaluate(model: TransformerLM, data: npt.NDArray[np.integer], args: argparse.Namespace, device: str) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(args.eval_iters):
        inputs, targets = get_batch(data, args.batch_size, args.context_length, device)
        total_loss += cross_entropy(model(inputs), targets).item()
    model.train()
    return total_loss / args.eval_iters


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = args.device or default_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = load_tokens(args.train_path, args.data_dtype)
    valid_data = load_tokens(args.valid_path, args.data_dtype)
    print(f"Train tokens: {len(train_data):,}, validation tokens: {len(valid_data):,}, device: {device}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"Model parameters: {num_params:,}")

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from {args.checkpoint_path} at iteration {start_iteration}")

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    for iteration in range(start_iteration, args.max_iters):
        lr = get_lr_cosine_schedule(iteration, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        loss = cross_entropy(model(inputs), targets)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Iterations are counted as completed steps, so a checkpoint saved at step n resumes at step n
        step = iteration + 1
        is_last_step = step == args.max_iters

        if step % args.log_interval == 0 or is_last_step:
            elapsed = time.time() - start_time
            print(f"step {step}/{args.max_iters} | train loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        if step % args.eval_interval == 0 or is_last_step:
            valid_loss = evaluate(model, valid_data, args, device)
            print(f"step {step}/{args.max_iters} | validation loss {valid_loss:.4f}")

        if step % args.checkpoint_interval == 0 or is_last_step:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
