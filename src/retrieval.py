"""Hybrid retrieval: fuse BM25 keyword search with vector search.

Vector search nails semantic similarity but misses exact terms (names, acronyms,
codes); BM25 nails exact terms but misses paraphrase. Reciprocal Rank Fusion (RRF)
combines their rankings without needing to normalise incompatible score scales.
"""
import re

import numpy as np

import config
import vectorstore

_bm25 = None         # cached BM25 index (rebuilding per query would be wasteful)
_bm25_chunks = None  # chunk list aligned to the BM25 corpus order


def _tokenize(text: str) -> list[str]:
    # Lowercase alnum tokens; keeps BM25 matching case- and punctuation-insensitive
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25():
    """Build (once) a BM25 index over every chunk in the vector store."""
    global _bm25, _bm25_chunks
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        _bm25_chunks = vectorstore.all_chunks()
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _bm25_chunks])
    return _bm25, _bm25_chunks


def _bm25_candidates(query: str, n: int) -> list[dict]:
    """Top-n chunks by BM25 keyword score."""
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    n = min(n, len(scores))
    # argpartition finds the top-n in O(N) without sorting the rest, then we sort
    # just those n by score descending — cheaper than a full sort on a big corpus
    top = np.argpartition(scores, -n)[-n:]
    top = top[np.argsort(scores[top])[::-1]]
    return [chunks[i] for i in top]


def hybrid(query: str, query_vector: list[float], n: int, rrf_k: int = config.RRF_K) -> list[dict]:
    """Fuse vector + BM25 rankings with RRF, return the top-n fused chunks."""
    vec_hits = vectorstore.search(query_vector, k=n)
    bm25_hits = _bm25_candidates(query, n)

    # RRF: each list contributes 1/(rrf_k + rank) to a chunk, rank starting at 1.
    # A chunk both lists rank highly bubbles up; rrf_k damps the top-rank dominance.
    fused: dict[str, float] = {}
    info: dict[str, dict] = {}
    for rank, h in enumerate(vec_hits, start=1):
        fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (rrf_k + rank)
        info[h["id"]] = h  # vector hit carries similarity as "score"
    for rank, h in enumerate(bm25_hits, start=1):
        fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (rrf_k + rank)
        info.setdefault(h["id"], {**h, "score": None})  # BM25-only chunk has no similarity

    out = []
    for cid in sorted(fused, key=lambda c: fused[c], reverse=True)[:n]:
        h = dict(info[cid])
        h["fusion_score"] = round(fused[cid], 4)
        out.append(h)
    return out


def reset_cache() -> None:
    """Drop the cached BM25 index; call after re-indexing so it rebuilds."""
    global _bm25, _bm25_chunks
    _bm25 = _bm25_chunks = None
