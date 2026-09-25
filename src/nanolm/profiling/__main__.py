import argparse

from nanolm.profiling.profiler import FunctionKey, Profiler
from nanolm.tokenizer import Tokenizer

DEFAULT_INPUT_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]

# Main processes of the tokenizer training
TOKENIZER_FUNCTIONS: list[FunctionKey] = [
    ("tokenizer.py", "train"),
    ("pretokenizer.py", "pretokenize_file"),
    ("pretokenizer.py", "_pretokenize"),
    ("pair_index.py", "_build_pair_index"),
    ("pair_index.py", "_build_heap"),
    ("pair_index.py", "best_pair"),
    ("pair_index.py", "merge"),
    ("pair_index.py", "_merge_pair"),
    ("pair_index.py", "_push_new_pairs"),
    ("pair_index.py", "_add_pretoken_pairs"),
    ("pair_index.py", "_remove_pretoken_pairs"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the BPE tokenizer training with cProfile")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--special-tokens", nargs="*", default=DEFAULT_SPECIAL_TOKENS)
    # cProfile only sees the main process: use 1 to profile pre-tokenization in-process
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--output-dir", default="profiles")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profiler = Profiler(output_dir=args.output_dir)
    tokenizer = Tokenizer(special_tokens=args.special_tokens)
    with profiler.track("train"):
        tokenizer.train(input_path=args.input_path, vocab_size=args.vocab_size, num_processes=args.num_processes)

    profiler.print_top("train", sort_by="tottime", top=args.top)
    profiler.print_breakdown("train", TOKENIZER_FUNCTIONS)
    print(f"\nLearned {len(tokenizer.merges)} merges. Profile saved to {profiler.profile_path('train')}")


if __name__ == "__main__":
    main()
