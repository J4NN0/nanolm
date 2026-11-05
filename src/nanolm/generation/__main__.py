import argparse

import torch

from nanolm.config import (
    CHECKPOINT_PATH,
    CONTEXT_LENGTH,
    D_FF,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    ROPE_THETA,
    TOKENIZER_PATH,
    VOCAB_SIZE,
)
from nanolm.generation import generate
from nanolm.tokenizer import Tokenizer
from nanolm.transformer import TransformerLM

DEFAULT_END_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer language model")

    parser.add_argument("--prompt", required=True)

    # Sampling
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--end-token", default=DEFAULT_END_TOKEN)

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

    tokenizer = Tokenizer.load(TOKENIZER_PATH)

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
    # Generation needs only the weights, not the optimizer state
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
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
