"""
oracle_pure.py — Pure, dependency-free Oracle functions extracted for testing.

This module re-exports the pure algorithmic functions from the Vibe Backend
Oracle without triggering the ChromaDB/SentenceTransformer initialization at
import time. Used exclusively by the system stress test harness.
"""

from __future__ import annotations

from typing import List


CONFIDENCE_THRESHOLD: float = 0.45
OVERFLOW_STRATEGIES = ("force_split", "truncate", "passthrough")


def chunk_intelligence(
    text: str,
    max_chars: int = 800,
    overflow_strategy: str = "force_split",
) -> List[str]:
    """Slice and dice raw web intelligence into semantic blocks.

    Splitting priority:
      1. Respects \\n line boundaries.
      2. If a line exceeds max_chars, splits at word (space) boundaries.
      3. If a sub-token still exceeds max_chars (no spaces — base64, minified JS,
         binary blobs), applies the overflow_strategy:
           'force_split'  — hard byte-boundary split. Guaranteed ≤ max_chars.
           'truncate'     — keep first max_chars bytes, discard the rest (lossy).
           'passthrough'  — original behaviour: emit the oversized token as-is.

    Args:
        text:              Input text to chunk.
        max_chars:         Soft target chunk size in characters.
        overflow_strategy: How to handle tokens wider than max_chars with no spaces.
    """
    if overflow_strategy not in OVERFLOW_STRATEGIES:
        raise ValueError(
            f"Unknown overflow_strategy {overflow_strategy!r}. Choose from {OVERFLOW_STRATEGIES}."
        )

    chunks: List[str] = []
    current_chunk = ""

    for line in text.split("\n"):
        # ── Word-boundary sub-split for lines wider than max_chars ────────────
        sub_lines: List[str] = []
        if max_chars > 0 and len(line) > max_chars:
            words = line.split(" ")
            sub = ""
            for word in words:
                if len(sub) + len(word) + 1 > max_chars and sub:
                    sub_lines.append(sub.strip())
                    sub = word + " "
                else:
                    sub += word + " "
            if sub.strip():
                sub_lines.append(sub.strip())
        else:
            sub_lines = [line]

        # ── Apply overflow_strategy to any token still too wide ───────────────
        final_sub_lines: List[str] = []
        for token in sub_lines:
            if max_chars > 0 and len(token) > max_chars:
                if overflow_strategy == "force_split":
                    # Hard byte-boundary split — guaranteed ≤ max_chars
                    while len(token) > max_chars:
                        final_sub_lines.append(token[:max_chars])
                        token = token[max_chars:]
                    if token:
                        final_sub_lines.append(token)
                elif overflow_strategy == "truncate":
                    final_sub_lines.append(token[:max_chars])
                else:  # passthrough — original behaviour
                    final_sub_lines.append(token)
            else:
                final_sub_lines.append(token)

        for sub_line in final_sub_lines:
            if max_chars > 0 and len(current_chunk) + len(sub_line) > max_chars:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sub_line + "\n"
            else:
                current_chunk += sub_line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks
