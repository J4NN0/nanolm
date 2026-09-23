from __future__ import annotations

import regex as re

# GPT-2 pre-tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

UTF8_ENCODING = "utf-8"
BYTE_LEVEL_TOKENIZER_VOCAB_SIZE = 256

Pair = tuple[bytes, bytes]


class Tokenizer:

    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self.special_tokens: list[str] = special_tokens or []
        self.vocab: dict[int, bytes] = self._init_base_vocabulary()
        self.merges: list[Pair] = []

    def _init_base_vocabulary(self) -> dict[int, bytes]:
        vocab: dict[int, bytes] = {}
        for byte in range(BYTE_LEVEL_TOKENIZER_VOCAB_SIZE):
            vocab[byte] = bytes([byte])
        for special_token in self.special_tokens:
            vocab[len(vocab)] = special_token.encode(UTF8_ENCODING)
        return vocab

    def train(self, input_path: str, vocab_size: int) -> None:
        if self.merges:
            raise RuntimeError("Tokenizer is already trained")

        with open(input_path) as f:
            text = f.read()

        pretoken_counts = self._pretokenize(text)
        words = self._init_word_symbols(pretoken_counts)

        while len(self.vocab) < vocab_size:
            pair_counts = self._count_pairs(words, pretoken_counts)
            if not pair_counts:
                break

            # Prefer the lexicographically greater pair
            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            self.merges.append(best_pair)

            new_token_id = len(self.vocab)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token_id] = new_token

            merged_words: dict[str, list[bytes]] = {}
            for pretoken, symbols in words.items():
                merged_words[pretoken] = self._merge_pair(symbols, best_pair)
            words = merged_words

    @staticmethod
    def _pretokenize(text: str) -> dict[str, int]:
        occ: dict[str, int] = {}
        tokens = re.findall(PAT, text)
        for token in tokens:
            occ[token] = occ.get(token, 0) + 1
        return occ

    @staticmethod
    def _init_word_symbols(pretoken_counts: dict[str, int]) -> dict[str, list[bytes]]:
        words: dict[str, list[bytes]] = {}
        for pretoken in pretoken_counts:
            encoded_pretoken = pretoken.encode(UTF8_ENCODING)
            symbols: list[bytes] = []
            for byte in encoded_pretoken:
                symbols.append(bytes([byte]))
            words[pretoken] = symbols
        return words

    @staticmethod
    def _count_pairs(
        words: dict[str, list[bytes]],
        pretoken_counts: dict[str, int],
    ) -> dict[Pair, int]:
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for pretoken, symbols in words.items():
            freq = pretoken_counts[pretoken]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        return pair_counts

    @staticmethod
    def _merge_pair(symbols: list[bytes], pair: Pair) -> list[bytes]:
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
