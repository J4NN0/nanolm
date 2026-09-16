from nanolm.tokenizer import train_bpe

DATA_PATH = "data/bpe_example.txt"


def main() -> None:
    pretoken_counts = train_bpe(
        input_path=DATA_PATH,
        vocab_size=0,
        special_tokens=[],
    )
    print(pretoken_counts)


if __name__ == "__main__":
    main()
