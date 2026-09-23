from __future__ import annotations

from nanolm.tokenizer.pretokenizer import UTF8_ENCODING

Pair = tuple[bytes, bytes]


class PairIndex:

    def __init__(self, pretoken_counts: dict[str, int]) -> None:
        self.pretoken_counts: dict[str, int] = pretoken_counts

        self.words: dict[str, list[bytes]] = self._init_word_symbols()
        self.pair_counts: dict[Pair, int] = {}
        self.pair_to_pretokens: dict[Pair, set[str]] = {}

        self._build_pair_index()

    def _init_word_symbols(self) -> dict[str, list[bytes]]:
        words: dict[str, list[bytes]] = {}
        for pretoken in self.pretoken_counts:
            encoded_pretoken = pretoken.encode(UTF8_ENCODING)
            symbols: list[bytes] = []
            for byte in encoded_pretoken:
                symbols.append(bytes([byte]))
            words[pretoken] = symbols
        return words

    def _build_pair_index(self) -> None:
        for pretoken in self.words:
            self._add_pretoken_pairs(pretoken)

    def best_pair(self) -> Pair | None:
        if not self.pair_counts:
            return None

        # Prefer the lexicographically greater pair
        return max(self.pair_counts, key=lambda pair: (self.pair_counts[pair], pair))

    def merge(self, pair: Pair) -> None:
        affected_pretokens = list(self.pair_to_pretokens[pair])
        for pretoken in affected_pretokens:
            self._remove_pretoken_pairs(pretoken)
            self.words[pretoken] = self._merge_pair(self.words[pretoken], pair)
            self._add_pretoken_pairs(pretoken)

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

    def _add_pretoken_pairs(self, pretoken: str) -> None:
        symbols = self.words[pretoken]
        freq = self.pretoken_counts[pretoken]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            self.pair_counts[pair] = self.pair_counts.get(pair, 0) + freq
            if pair not in self.pair_to_pretokens:
                self.pair_to_pretokens[pair] = set()
            self.pair_to_pretokens[pair].add(pretoken)

    def _remove_pretoken_pairs(self, pretoken: str) -> None:
        symbols = self.words[pretoken]
        freq = self.pretoken_counts[pretoken]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            self.pair_counts[pair] -= freq
            if self.pair_counts[pair] == 0:
                del self.pair_counts[pair]
                del self.pair_to_pretokens[pair]
            else:
                self.pair_to_pretokens[pair].discard(pretoken)
