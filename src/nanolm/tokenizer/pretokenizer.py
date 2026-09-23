from __future__ import annotations

import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO

# GPT-2 pre-tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

UTF8_ENCODING = "utf-8"
CHUNK_SIZE_BYTES = 50 * 1024 * 1024


class Pretokenizer:

    def __init__(self, special_tokens: list[str]) -> None:
        self.special_tokens: list[str] = special_tokens
        self.special_pattern: str | None = self._build_special_pattern()

    def _build_special_pattern(self) -> str | None:
        if not self.special_tokens:
            return None

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_special_tokens: list[str] = []
        for special_token in sorted_special_tokens:
            escaped_special_tokens.append(re.escape(special_token))

        return "|".join(escaped_special_tokens)

    def pretokenize_file(self, input_path: str, num_processes: int) -> dict[str, int]:
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
            boundaries = self._find_chunk_boundaries(f, desired_num_chunks, split_special_token)

        chunk_ranges: list[tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            chunk_ranges.append((boundaries[i], boundaries[i + 1]))
        return chunk_ranges

    # Adapted from Stanford CS336's `pretokenization_example.py`:
    # https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
    # Copyright 2025 Stanford University. Licensed under the MIT License.
    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _pretokenize_chunk(self, input_path: str, start: int, end: int) -> dict[str, int]:
        chunk_text = self._read_chunk(input_path, start, end)
        return self._pretokenize(chunk_text)

    @staticmethod
    def _read_chunk(input_path: str, start: int, end: int) -> str:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)

        return chunk.decode(UTF8_ENCODING, errors="ignore")

    def _pretokenize(self, text: str) -> dict[str, int]:
        if self.special_pattern is not None:
            segments = re.split(self.special_pattern, text)
        else:
            segments = [text]

        occ: dict[str, int] = {}
        for segment in segments:
            for match in re.finditer(PAT, segment):
                token = match.group()
                occ[token] = occ.get(token, 0) + 1

        return occ
