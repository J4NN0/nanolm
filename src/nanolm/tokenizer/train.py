import regex as re

# GPT-2 pre-tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

UTF8_ENCODING = "utf-8"
BYTE_LEVEL_TOKENIZER_VOCAB_SIZE = 256

def pretokenize(text: str) -> dict[str, int]:
    occ: dict[str, int] = {}
    tokens = re.findall(PAT, text)
    for token in tokens:
        occ[token] = occ.get(token, 0) + 1
    return occ


def init_word_symbols(pretoken_counts: dict[str, int]) -> dict[str, list[bytes]]:
    words: dict[str, list[bytes]] = {}
    for pretoken in pretoken_counts:
        encoded_pretoken = pretoken.encode(UTF8_ENCODING)
        symbols: list[bytes] = []
        for byte in encoded_pretoken:
            symbols.append(bytes([byte]))
        words[pretoken] = symbols
    return words


def init_base_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for byte in range(BYTE_LEVEL_TOKENIZER_VOCAB_SIZE):
        vocab[byte] = bytes([byte])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode(UTF8_ENCODING)
    return vocab


def count_pairs(
    words: dict[str, list[bytes]],
    pretoken_counts: dict[str, int],
) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for pretoken, symbols in words.items():
        freq = pretoken_counts[pretoken]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def merge_pair(symbols: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    merged: list[bytes] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path) as f:
        text = f.read()

    pretoken_counts = pretokenize(text)
    words = init_word_symbols(pretoken_counts)
    vocab = init_base_vocabulary(special_tokens)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counts = count_pairs(words, pretoken_counts)
        if not pair_counts:
            break

        # Prefer the lexicographically greater pair
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)

        new_token_id = len(vocab)
        new_token = best_pair[0] + best_pair[1]
        vocab[new_token_id] = new_token

        merged_words: dict[str, list[bytes]] = {}
        for pretoken, symbols in words.items():
            merged_words[pretoken] = merge_pair(symbols, best_pair)
        words = merged_words

    return vocab, merges
