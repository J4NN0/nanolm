import argparse

import torch

from nanolm.generation import generate
from nanolm.steering import Steering, steering_vector
from nanolm.tokenizer import Tokenizer
from nanolm.transformer import TransformerLM

DEFAULT_CHECKPOINT_PATH = "checkpoints/TinyStoriesV2-GPT4_10000.pt"
DEFAULT_TOKENIZER_PATH = "tokenizers/TinyStoriesV2-GPT4_10000.json"
DEFAULT_END_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text while steering the model towards a concept")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--toward", required=True, help='Word or text with the concept to steer towards, e.g. "dragon"')
    parser.add_argument(
        "--away-from",
        default=None,
        help='Word or text with the concept to steer away from, e.g. "girl" (default: none, only --toward is used)',
    )
    parser.add_argument("--layer", type=int, default=1, help="Block after which the vector is added (from 1)")
    parser.add_argument("--strength", type=float, default=1.0, help="Multiplier of the vector; 0 disables steering")
    parser.add_argument(
        "--every-position",
        action="store_true",
        help="Add the vector at every position, not only the prompt: a stronger push, less natural text (try --strength 0.5)",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--compare", action="store_true", help="Also print the same samples without steering")

    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)

    # Sampling
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--end-token", default=DEFAULT_END_TOKEN)

    # Model: must match the trained checkpoint
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--device", default=None, help="e.g. cpu, cuda:0, mps (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_concept(tokenizer: Tokenizer, text: str) -> list[int]:
    # Inside a story a word follows a space, and "␣dragon" is a different token from "dragon",
    # so a single word gets the space it would have in the text
    if " " not in text:
        text = " " + text
    return tokenizer.encode(text)


def main() -> None:
    args = parse_args()
    device = args.device or default_device()

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
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    end_token_id = None
    end_token = args.end_token.encode("utf-8")
    if end_token in tokenizer.token_ids:
        end_token_id = tokenizer.token_ids[end_token]

    toward_ids = encode_concept(tokenizer, args.toward)
    away_ids = None
    if args.away_from is not None:
        away_ids = encode_concept(tokenizer, args.away_from)

    vector = steering_vector(model, toward_ids, away_ids, args.layer)
    print(f"Steering vector after block {args.layer}: length {vector.norm().item():.1f}, strength {args.strength}")

    prompt_ids = tokenizer.encode(args.prompt)
    # By default only the prompt is steered, and the new tokens read it through attention
    num_positions = len(prompt_ids)
    if args.every_position:
        num_positions = None

    for sample in range(args.num_samples):
        # Same seed for both versions of a sample, so the only difference between them is the steering
        seed = args.seed + sample

        if args.compare:
            torch.manual_seed(seed)
            new_token_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, end_token_id)
            print(f"\n--- sample {sample + 1}, not steered ---")
            print(args.prompt + tokenizer.decode(new_token_ids))

        torch.manual_seed(seed)
        with Steering(model, args.layer, vector, args.strength, num_positions):
            new_token_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, end_token_id)
        print(f"\n--- sample {sample + 1}, steered ---")
        print(args.prompt + tokenizer.decode(new_token_ids))


if __name__ == "__main__":
    main()
