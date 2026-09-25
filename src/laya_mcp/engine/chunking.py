"""Token-aware text chunking with character spans.

Used by `laya_scan_untrusted` (and anything else that has to feed long text to a 512-token
checkpoint). Chunks carry the character span they came from so callers can point at the exact
region of the original text.

Two strategies:
- HF *fast* tokenizers expose `offset_mapping`, so windows are cut on exact token boundaries.
- Anything else (slow tokenizers, test fakes, no tokenizer at all) falls back to whitespace
  words, with the tokens-per-word ratio estimated from a sample of the text.
"""
from __future__ import annotations

import re
from typing import Any, NamedTuple

_WORD = re.compile(r"\S+")
_DEFAULT_TOKENS_PER_WORD = 1.3   # English BPE average, used when no tokenizer is available
_SAMPLE_WORDS = 2000


class Chunk(NamedTuple):
    start: int   # character offset in the original text (inclusive)
    end: int     # character offset in the original text (exclusive)
    text: str    # == original[start:end]


def chunk_text(tokenizer: Any, text: str, max_tokens: int = 350, overlap: int = 40) -> list[Chunk]:
    """Split `text` into windows of at most ~`max_tokens` tokens overlapping by `overlap` tokens.

    Returns [] for empty/whitespace-only text and a single chunk covering the whole text when it
    already fits.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap must be >= 0 and smaller than max_tokens")
    if not text or not text.strip():
        return []

    offsets = _offsets(tokenizer, text)
    if offsets is not None:
        return _windows(text, offsets, max_tokens, overlap)
    return _word_chunks(tokenizer, text, max_tokens, overlap)


# -- fast tokenizer path ----------------------------------------------------------------------

def _offsets(tokenizer: Any, text: str) -> list[tuple[int, int]] | None:
    """Character offsets per token, or None when the tokenizer cannot provide them."""
    if tokenizer is None or not callable(tokenizer):
        return None
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:  # slow tokenizers raise NotImplementedError; fakes may reject kwargs
        return None
    try:
        raw = enc["offset_mapping"]
    except (KeyError, TypeError):
        return None
    if raw and isinstance(raw[0], (list, tuple)) and raw[0] and isinstance(raw[0][0], (list, tuple)):
        raw = raw[0]   # batched output
    offsets = [(int(s), int(e)) for s, e in raw if e > s]
    return offsets or None


def _windows(text: str, offsets: list[tuple[int, int]], max_tokens: int, overlap: int) -> list[Chunk]:
    n = len(offsets)
    if n <= max_tokens:
        start, end = _trim(text, 0, len(text))
        return [Chunk(start, end, text[start:end])]
    step = max_tokens - overlap
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        start, end = offsets[i][0], offsets[j - 1][1]
        if j == n:
            end = len(text)
        start, end = _trim(text, start, end)
        if end > start:
            chunks.append(Chunk(start, end, text[start:end]))
        if j == n:
            break
        i += step
    return chunks


# -- word fallback ----------------------------------------------------------------------------

def _tokens_per_word(tokenizer: Any, words: list[str]) -> float:
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return _DEFAULT_TOKENS_PER_WORD
    sample = words[:_SAMPLE_WORDS]
    try:
        n_tok = len(tokenizer.encode(" ".join(sample), add_special_tokens=False))
    except TypeError:
        n_tok = len(tokenizer.encode(" ".join(sample)))
    except Exception:
        return _DEFAULT_TOKENS_PER_WORD
    return max(n_tok / max(len(sample), 1), 0.1)


def _word_chunks(tokenizer: Any, text: str, max_tokens: int, overlap: int) -> list[Chunk]:
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    words = [text[s:e] for s, e in spans]
    ratio = _tokens_per_word(tokenizer, words)
    per_chunk = max(1, int(max_tokens / ratio))
    overlap_words = min(int(overlap / ratio), per_chunk - 1)
    if len(spans) <= per_chunk:
        start, end = spans[0][0], spans[-1][1]
        return [Chunk(start, end, text[start:end])]
    step = per_chunk - overlap_words
    chunks: list[Chunk] = []
    i = 0
    while i < len(spans):
        j = min(i + per_chunk, len(spans))
        start, end = spans[i][0], spans[j - 1][1]
        chunks.append(Chunk(start, end, text[start:end]))
        if j == len(spans):
            break
        i += step
    return chunks


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
