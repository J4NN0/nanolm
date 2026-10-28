import argparse

import torch

from nanolm.generation import generate
from nanolm.tokenizer import Tokenizer
from nanolm.transformer import TransformerLM

DEFAULT_CHECKPOINT_PATH = "checkpoints/checkpoint.pt"
DEFAULT_END_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer language model")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)

    # Sampling
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--end-token", default=DEFAULT_END_TOKEN)

    # Model: must match the values used for training
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--device", default=None, help="e.g. cpu, cuda:0, mps (default: best available)")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    device = args.device or default_device()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = Tokenizer.load(args.tokenizer_path)

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
    # Generation needs only the weights, not the optimizer state
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    end_token_id = None
    end_token = args.end_token.encode("utf-8")
    if end_token in tokenizer.token_ids:
        end_token_id = tokenizer.token_ids[end_token]

    prompt_ids = tokenizer.encode(args.prompt)
    new_token_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, end_token_id)
    print(args.prompt + tokenizer.decode(new_token_ids))


if __name__ == "__main__":
    main()
