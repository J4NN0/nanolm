from nanolm.tokenizer import Tokenizer

DATA_PATH = "data/bpe_example.txt"
VOCAB_SIZE = 256 + 6


def main() -> None:
    tokenizer = Tokenizer(special_tokens=[])
    tokenizer.train(input_path=DATA_PATH, vocab_size=VOCAB_SIZE)
    print("Merges:")
    for token1, token2 in tokenizer.merges:
        print(f"  {token1!r} + {token2!r} -> {(token1 + token2)!r}")


if __name__ == "__main__":
    main()
