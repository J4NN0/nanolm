from __future__ import annotations

import heapq
from nanolm.tokenizer.pretokenizer import UTF8_ENCODING

Pair = tuple[bytes, bytes]


class ReversedPair:
    __slots__ = ("pair",)

    def __init__(self, pair: Pair) -> None:
        self.pair: Pair = pair

    def __lt__(self, other: ReversedPair) -> bool:
        return self.pair > other.pair


HeapEntry = tuple[int, ReversedPair]


class PairIndex:

    def __init__(self, pretoken_counts: dict[str, int]) -> None:
        self.pretoken_counts: dict[str, int] = pretoken_counts

        self.words: dict[str, list[bytes]] = self._init_word_symbols()
        self.pair_counts: dict[Pair, int] = {}
        self.pair_to_pretokens: dict[Pair, set[str]] = {}
        self.heap: list[HeapEntry] = []

        self._build_pair_index()
        self._build_heap()

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

    def _build_heap(self) -> None:
        for pair, count in self.pair_counts.items():
            self.heap.append((-count, ReversedPair(pair)))
        heapq.heapify(self.heap)

    def best_pair(self) -> Pair | None:
        while self.heap:
            neg_count, reversed_pair = self.heap[0]
            pair = reversed_pair.pair
            count = self.pair_counts.get(pair, 0)
            if count == -neg_count:
                return pair

            # Stale entry: re-insert the pair with its current (lower) count, or drop it if it no longer exists
            if count > 0:
                heapq.heapreplace(self.heap, (-count, reversed_pair))
            else:
                heapq.heappop(self.heap)

        return None

    def merge(self, pair: Pair) -> None:
        affected_pretokens = list(self.pair_to_pretokens[pair])
        for pretoken in affected_pretokens:
            self._remove_pretoken_pairs(pretoken)
            self.words[pretoken] = self._merge_pair(self.words[pretoken], pair)
            self._add_pretoken_pairs(pretoken)

        self._push_new_pairs(pair, affected_pretokens)

    def _push_new_pairs(self, pair: Pair, affected_pretokens: list[str]) -> None:
        # Only pairs containing the merged token can have a higher count than their heap entry
        merged_token = pair[0] + pair[1]
        new_pairs: set[Pair] = set()

        for pretoken in affected_pretokens:
            symbols = self.words[pretoken]
            for i in range(len(symbols) - 1):
                if symbols[i] == merged_token or symbols[i + 1] == merged_token:
                    new_pairs.add((symbols[i], symbols[i + 1]))

        for new_pair in new_pairs:
            heapq.heappush(self.heap, (-self.pair_counts[new_pair], ReversedPair(new_pair)))

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
