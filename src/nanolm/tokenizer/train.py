import regex as re

# GPT-2 pre-tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> dict[str, int]:
    occ: dict[str, int] = {}
    tokens = re.findall(PAT, text)
    for token in tokens:
        occ[token] = occ.get(token, 0) + 1
    return occ


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> dict[str, int]:
    """Given the path to an input corpus, train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        dict[str, int]: Pre-token counts for the corpus at `input_path`.
            This is currently the only step implemented; the pair-counting and
            merging steps that turn this into a vocabulary and merge list are
            built on top of this next.
    """
    with open(input_path) as f:
        text = f.read()
    return pretokenize(text)
