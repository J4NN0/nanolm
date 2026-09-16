from nanolm.tokenizer import train_bpe

DATA_PATH = "data/bpe_example.txt"
VOCAB_SIZE = 256 + 6


def main() -> None:
    vocab, merges = train_bpe(
        input_path=DATA_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=[],
    )
    print("Merges:")
    for token1, token2 in merges:
        print(f"  {token1!r} + {token2!r} -> {(token1 + token2)!r}")


if __name__ == "__main__":
    main()
