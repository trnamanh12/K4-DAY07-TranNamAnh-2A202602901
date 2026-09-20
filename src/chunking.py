from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return []
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def _merge_splits(self, splits: list[str], sep: str) -> list[str]:
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            _len = len(d)
            add_len = _len + (len(sep) if current_doc else 0)
            if total + add_len <= self.chunk_size:
                current_doc.append(d)
                total += add_len
            else:
                if current_doc:
                    doc = sep.join(current_doc)
                    if doc:
                        docs.append(doc)
                    current_doc = []
                    total = 0
                current_doc.append(d)
                total = _len
        if current_doc:
            doc = sep.join(current_doc)
            if doc:
                docs.append(doc)
        return docs

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = None
        sep_idx = -1
        for i, s in enumerate(remaining_separators):
            if s == "" or s in current_text:
                sep = s
                sep_idx = i
                break

        if sep is None:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        next_separators = remaining_separators[sep_idx + 1 :]
        if sep == "":
            splits = list(current_text)
        else:
            splits = current_text.split(sep)

        good_splits: list[str] = []
        final_chunks: list[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, sep)
                    final_chunks.extend(merged)
                    good_splits = []
                if not next_separators:
                    final_chunks.extend([s[i : i + self.chunk_size] for i in range(0, len(s), self.chunk_size)])
                else:
                    other_chunks = self._split(s, next_separators)
                    final_chunks.extend(other_chunks)
        if good_splits:
            merged = self._merge_splits(good_splits, sep)
            final_chunks.extend(merged)

        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    dot = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=min(50, max(0, chunk_size // 10)))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        strategies = {
            "fixed_size": fixed_chunker.chunk(text),
            "by_sentences": sentence_chunker.chunk(text),
            "recursive": recursive_chunker.chunk(text),
        }

        results = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_len = (sum(len(c) for c in chunks) / count) if count > 0 else 0.0
            results[name] = {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks,
            }
        return results
