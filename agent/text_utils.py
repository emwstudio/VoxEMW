"""Text splitting helpers for TTS.

Pure-logic module with zero heavy dependencies (stdlib only),
so it can be unit-tested on machines without GPU dependencies.
"""

from __future__ import annotations

import re

#: Default maximum number of characters fed to the TTS in one go.
DEFAULT_MAX_CHARS = 200

#: Clause-ending punctuation (Chinese + ASCII) used as preferred split points.
_CLAUSE_RE = re.compile(r"[^。！？；，、.!?;,\n]+[。！？；，、.!?;,\n]*")


def split_text_for_tts(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Splitting prefers sentence/clause boundaries (。！？；，、 and ASCII
    equivalents). A single clause longer than *max_chars* is hard-split.
    Returns an empty list for empty/whitespace-only input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for match in _CLAUSE_RE.finditer(text):
        clause = match.group(0)
        # Hard-split clauses that alone exceed the limit.
        while len(clause) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(clause[:max_chars])
            clause = clause[max_chars:]

        if current and len(current) + len(clause) > max_chars:
            chunks.append(current)
            current = clause
        else:
            current += clause

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk.strip()]
