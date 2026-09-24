from __future__ import annotations

import cProfile
import os
import pstats
import time
from contextlib import contextmanager
from typing import Iterator

DEFAULT_OUTPUT_DIR = "profiles"

# (file name suffix, function name), e.g. ("pair_index.py", "merge")
FunctionKey = tuple[str, str]


class Profiler:

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir: str = output_dir
        self.stats: dict[str, pstats.Stats] = {}
        self.wall_times: dict[str, float] = {}

    @contextmanager
    def track(self, label: str) -> Iterator[None]:
        profile = cProfile.Profile()
        start = time.perf_counter()
        profile.enable()
        try:
            yield
        finally:
            profile.disable()
            self.wall_times[label] = time.perf_counter() - start
            self.stats[label] = pstats.Stats(profile)
            self._dump(label)

    def _dump(self, label: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.stats[label].dump_stats(self.profile_path(label))

    def profile_path(self, label: str) -> str:
        return os.path.join(self.output_dir, f"{label}.prof")

    def print_top(self, label: str, sort_by: str = "cumulative", top: int = 20) -> None:
        print(f"\n=== {label}: top {top} functions by {sort_by} time ===")
        self.stats[label].sort_stats(sort_by).print_stats(top)

    def print_breakdown(self, label: str, functions: list[FunctionKey]) -> None:
        wall_time = self.wall_times[label]
        print(f"\n=== {label}: breakdown (wall time {wall_time:.3f}s) ===")
        print(f"{'function':<40}{'calls':>12}{'tottime (s)':>14}{'cumtime (s)':>14}{'% wall':>9}")

        for file_suffix, function_name in functions:
            entry = self._find_function(label, file_suffix, function_name)
            name = f"{file_suffix}:{function_name}"
            if entry is None:
                print(f"{name:<40}{'not called in this process':>49}")
                continue

            calls, tottime, cumtime = entry
            share = 100 * cumtime / wall_time
            print(f"{name:<40}{calls:>12}{tottime:>14.3f}{cumtime:>14.3f}{share:>8.1f}%")

    def _find_function(self, label: str, file_suffix: str, function_name: str) -> tuple[int, float, float] | None:
        # pstats keys entries by (filename, line number, function name) and
        # stores (primitive calls, total calls, tottime, cumtime, callers)
        for (filename, _, name), (_, calls, tottime, cumtime, _) in self.stats[label].stats.items():
            if name == function_name and filename.endswith(file_suffix):
                return calls, tottime, cumtime
        return None
