"""
retriever.py — Corpus loader and TF-IDF retriever.

Loads all markdown files from data/{hackerrank,claude,visa}/,
builds a TF-IDF index, and exposes retrieve(query, top_k) that
returns the most relevant document chunks with their full text.
"""

from __future__ import annotations

import math
import pathlib
import re
from collections import defaultdict
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths & tuning
# ---------------------------------------------------------------------------

DATA_ROOT = pathlib.Path(__file__).parent.parent / "data"
CHUNK_SIZE = 1200       # characters per chunk
CHUNK_OVERLAP = 300     # overlap between consecutive chunks
TOP_K_DEFAULT = 6


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _load_markdown_files() -> List[dict]:
    docs = []
    for md_path in DATA_ROOT.rglob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parts = md_path.relative_to(DATA_ROOT).parts
        domain = parts[0] if parts else "unknown"
        title = md_path.stem.replace("-", " ").replace("_", " ")
        docs.append({"path": str(md_path), "domain": domain, "title": title, "text": text})
    return docs


def _chunk_document(doc: dict) -> List[dict]:
    text = doc["text"]
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({**doc, "text": text[start:end]})
        if end >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "it", "its", "i", "my", "we", "our", "you", "your", "he", "she",
    "they", "their", "not", "no", "if", "as", "so", "up", "out", "about",
    "what", "how", "when", "where", "who", "which", "all", "any", "more",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1 and t not in _STOPWORDS]


def _build_tfidf(chunks: List[dict]):
    n = len(chunks)
    tf_matrix: List[dict] = []
    df: dict = defaultdict(int)

    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        tf: dict = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total = sum(tf.values()) or 1
        tf = {t: v / total for t, v in tf.items()}
        tf_matrix.append(tf)
        for t in tf:
            df[t] += 1

    idf = {t: math.log((n + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}
    return tf_matrix, idf


def _score(query_tokens: List[str], tf: dict, idf: dict) -> float:
    return sum(tf.get(t, 0) * idf.get(t, 0) for t in query_tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Retriever:
    _instance: "Retriever | None" = None

    def __init__(self) -> None:
        print("Loading corpus …", flush=True)
        raw_docs = _load_markdown_files()
        print(f"  {len(raw_docs)} documents loaded", flush=True)
        self._chunks: List[dict] = []
        for doc in raw_docs:
            self._chunks.extend(_chunk_document(doc))
        print(f"  {len(self._chunks)} chunks indexed", flush=True)
        self._tf_matrix, self._idf = _build_tfidf(self._chunks)
        print("  TF-IDF index ready\n", flush=True)

    @classmethod
    def get(cls) -> "Retriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        domain_filter: str | None = None,
    ) -> List[dict]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores: List[Tuple[float, int]] = []
        for idx, tf in enumerate(self._tf_matrix):
            chunk = self._chunks[idx]
            if domain_filter and chunk["domain"] != domain_filter:
                continue
            s = _score(tokens, tf, self._idf)
            if s > 0:
                scores.append((s, idx))

        scores.sort(reverse=True)
        results, seen = [], set()
        for s, idx in scores:
            chunk = self._chunks[idx]
            if chunk["path"] in seen:
                continue
            seen.add(chunk["path"])
            results.append({**chunk, "score": s})
            if len(results) >= top_k:
                break
        return results


def retrieve(query: str, top_k: int = TOP_K_DEFAULT, domain_filter: str | None = None) -> List[dict]:
    return Retriever.get().retrieve(query, top_k=top_k, domain_filter=domain_filter)
