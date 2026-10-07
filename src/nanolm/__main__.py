import os

from nanolm.tokenizer import Tokenizer

DATA_PATH = "data/bpe_example.txt"
VOCAB_SIZE = 256 + 6
SPECIAL_TOKENS: list[str] = []

TOKENIZERS_DIR = "tokenizers"
DATASET_NAME = os.path.splitext(os.path.basename(DATA_PATH))[0]
TOKENIZER_PATH = os.path.join(TOKENIZERS_DIR, f"{DATASET_NAME}_{VOCAB_SIZE}.json")


def load_or_train_tokenizer() -> Tokenizer:
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH} ...")
        return Tokenizer.load(TOKENIZER_PATH)

    tokenizer = Tokenizer(special_tokens=SPECIAL_TOKENS)
    tokenizer.train(input_path=DATA_PATH, vocab_size=VOCAB_SIZE)
    tokenizer.save(TOKENIZER_PATH)

    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    return tokenizer


def main() -> None:
    tokenizer = load_or_train_tokenizer()
    print("Merges:")
    for token1, token2 in tokenizer.merges:
        print(f"  {token1!r} + {token2!r} -> {(token1 + token2)!r}")


if __name__ == "__main__":
    main()
