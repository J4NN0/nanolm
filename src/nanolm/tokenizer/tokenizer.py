from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from nanolm.tokenizer.pair_index import Pair, PairIndex
from nanolm.tokenizer.pretokenizer import UTF8_ENCODING, Pretokenizer

BYTE_LEVEL_TOKENIZER_VOCAB_SIZE = 256

# Token IDs are stored on disk as 2-byte integers, which fit vocabularies of up to 65,536 tokens
TOKEN_ID_DTYPE = np.uint16

# Maps each byte (0-255) to exactly one character, so tokens can be stored as JSON strings losslessly
TOKEN_SERIALIZATION_ENCODING = "latin-1"


class Tokenizer:

    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self.special_tokens: list[str] = special_tokens or []

        self.vocab: dict[int, bytes] = self._init_base_vocabulary()
        self.merges: list[Pair] = []

        self.pretokenizer = Pretokenizer(self.special_tokens)
        self.token_ids: dict[bytes, int] = {}
        self.merge_ranks: dict[Pair, int] = {}
        self.pretoken_cache: dict[str, list[int]] = {}
        self._build_encoding_tables()

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

        pretoken_counts = Pretokenizer(self.special_tokens).pretokenize_file(input_path, num_processes)
        pair_index = PairIndex(pretoken_counts)

        while len(self.vocab) < vocab_size:
            best_pair = pair_index.best_pair()
            if best_pair is None:
                break

            self.merges.append(best_pair)
            self._add_new_token(best_pair)

            pair_index.merge(best_pair)

        self._build_encoding_tables()

    def _add_new_token(self, best_pair: Pair):
        new_token_id = len(self.vocab)
        new_token = best_pair[0] + best_pair[1]
        self.vocab[new_token_id] = new_token

    def _build_encoding_tables(self) -> None:
        # Reverse of the vocabulary: token bytes -> token ID
        self.token_ids = {}
        for token_id, token in self.vocab.items():
            self.token_ids[token] = token_id

        # Position of every merge in the learned order: a lower rank was learned earlier
        self.merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks[pair] = rank

        # Same pre-token always encodes the same way, so each one is encoded once
        self.pretoken_cache = {}

    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenizer.split_text(text)

        token_ids: list[int] = []
        for pretoken in pretokens:
            if pretoken in self.special_tokens:
                token_ids.append(self.token_ids[pretoken.encode(UTF8_ENCODING)])
            else:
                token_ids.extend(self._encode_pretoken(pretoken))
        return token_ids

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        if pretoken in self.pretoken_cache:
            return self.pretoken_cache[pretoken]

        symbols: list[bytes] = []
        for byte in pretoken.encode(UTF8_ENCODING):
            symbols.append(bytes([byte]))

        # Apply the merges in the order they were learned: always merge the earliest-learned pair present
        while len(symbols) > 1:
            best_pair: Pair | None = None
            best_rank = len(self.merges)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair, best_rank)
                if rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break
            symbols = PairIndex._merge_pair(symbols, best_pair)

        token_ids: list[int] = []
        for symbol in symbols:
            token_ids.append(self.token_ids[symbol])

        self.pretoken_cache[pretoken] = token_ids
        return token_ids

    def encode_file(self, input_path: str, output_path: str, num_processes: int | None = None) -> int:
        if len(self.vocab) > np.iinfo(TOKEN_ID_DTYPE).max + 1:
            raise ValueError(f"Vocabulary of {len(self.vocab)} tokens does not fit in {np.dtype(TOKEN_ID_DTYPE).name}")

        if num_processes is None:
            num_processes = os.cpu_count() or 1

        # Chunks of about 50 MB that end at a document separator, so no pre-token is cut in two
        chunk_ranges = self.pretokenizer._find_chunk_ranges(input_path, num_processes)

        tasks: list[tuple[str, int, int]] = []
        for start, end in chunk_ranges:
            tasks.append((input_path, start, end))

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Chunks are encoded in parallel, but written in file order, one at a time,
        # so the whole token list never has to be in memory
        num_tokens = 0
        with open(output_path, "wb") as f, Pool(min(num_processes, len(tasks))) as pool:
            for chunk_token_ids in pool.imap(self._encode_chunk, tasks):
                chunk_token_ids.tofile(f)
                num_tokens += len(chunk_token_ids)
        return num_tokens

    def _encode_chunk(self, task: tuple[str, int, int]) -> np.ndarray:
        input_path, start, end = task
        chunk_text = Pretokenizer._read_chunk(input_path, start, end)
        return np.array(self.encode(chunk_text), dtype=TOKEN_ID_DTYPE)

    def decode(self, token_ids: list[int]) -> str:
        # Joined once at the end: adding to a bytes object copies it every time
        tokens: list[bytes] = []
        for token_id in token_ids:
            tokens.append(self.vocab[token_id])
        return b"".join(tokens).decode(UTF8_ENCODING, errors="replace")

    def save(self, path: str) -> None:
        vocab: dict[str, str] = {}
        for token_id, token in self.vocab.items():
            vocab[str(token_id)] = token.decode(TOKEN_SERIALIZATION_ENCODING)

        merges: list[list[str]] = []
        for token1, token2 in self.merges:
            merges.append([token1.decode(TOKEN_SERIALIZATION_ENCODING), token2.decode(TOKEN_SERIALIZATION_ENCODING)])

        data = {
            "special_tokens": self.special_tokens,
            "vocab": vocab,
            "merges": merges,
        }

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding=UTF8_ENCODING) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, encoding=UTF8_ENCODING) as f:
            data = json.load(f)

        tokenizer = cls(special_tokens=data["special_tokens"])

        vocab: dict[int, bytes] = {}
        for token_id, token in data["vocab"].items():
            vocab[int(token_id)] = token.encode(TOKEN_SERIALIZATION_ENCODING)
        tokenizer.vocab = vocab

        merges: list[Pair] = []
        for token1, token2 in data["merges"]:
            merges.append((token1.encode(TOKEN_SERIALIZATION_ENCODING), token2.encode(TOKEN_SERIALIZATION_ENCODING)))
        tokenizer.merges = merges

        tokenizer._build_encoding_tables()
        return tokenizer
