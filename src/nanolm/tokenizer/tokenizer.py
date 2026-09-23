from __future__ import annotations

import os
import regex as re
from multiprocessing import Pool
from nanolm.tokenizer.pretokenization import find_chunk_boundaries

# GPT-2 pre-tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

UTF8_ENCODING = "utf-8"
BYTE_LEVEL_TOKENIZER_VOCAB_SIZE = 256
CHUNK_SIZE_BYTES = 50 * 1024 * 1024

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

    def train(self, input_path: str, vocab_size: int, num_processes: int | None = None) -> None:
        if self.merges:
            raise RuntimeError("Tokenizer is already trained")

        if num_processes is None:
            num_processes = os.cpu_count() or 1

        pretoken_counts = self._parallel_pretokenize(input_path, num_processes)
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

    def _parallel_pretokenize(self, input_path: str, num_processes: int) -> dict[str, int]:
        chunk_ranges = self._find_chunk_ranges(input_path, num_processes)

        tasks: list[tuple[str, int, int]] = []
        for start, end in chunk_ranges:
            tasks.append((input_path, start, end))

        if len(tasks) <= 1:
            chunk_counts: list[dict[str, int]] = []
            for task in tasks:
                chunk_counts.append(self._pretokenize_chunk(*task))
        else:
            with Pool(min(num_processes, len(tasks))) as pool:
                chunk_counts = pool.starmap(self._pretokenize_chunk, tasks)

        pretoken_counts: dict[str, int] = {}
        for counts in chunk_counts:
            for pretoken, count in counts.items():
                pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + count
        return pretoken_counts

    def _find_chunk_ranges(self, input_path: str, num_processes: int) -> list[tuple[int, int]]:
        file_size = os.path.getsize(input_path)

        if not self.special_tokens:
            return [(0, file_size)]

        # The first special token is assumed to be the document separator
        split_special_token = self.special_tokens[0].encode(UTF8_ENCODING)
        desired_num_chunks = max(num_processes, file_size // CHUNK_SIZE_BYTES)
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)

        chunk_ranges: list[tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            chunk_ranges.append((boundaries[i], boundaries[i + 1]))
        return chunk_ranges

    def _pretokenize_chunk(self, input_path: str, start: int, end: int) -> dict[str, int]:
        chunk_text = self._decode(input_path, start, end)
        return self._pretokenize(chunk_text)

    @staticmethod
    def _decode(input_path: str, start: int, end: int):
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)

        return chunk.decode(UTF8_ENCODING, errors="ignore")

    def _pretokenize(self, text: str) -> dict[str, int]:
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_special_tokens: list[str] = []
            for special_token in sorted_special_tokens:
                escaped_special_tokens.append(re.escape(special_token))

            special_pattern = "|".join(escaped_special_tokens)
            segments = re.split(special_pattern, text)
        else:
            segments = [text]

        occ: dict[str, int] = {}
        for segment in segments:
            for match in re.finditer(PAT, segment):
                token = match.group()
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
