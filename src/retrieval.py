# Hybrid Retrieval - BM25 + Vector, Fused With RRF
import re

import numpy as np

import config
import vectorstore

_bm25 = None         # Cached BM25 Index - Rebuilding per Query Would Be Wasteful
_bm25_chunks = None  # Chunk List Aligned to the BM25 Corpus Order


def _tokenize(text: str) -> list[str]:
    # Lowercase Alnum Tokens - Keeps BM25 Case- and Punctuation-Insensitive
    return re.findall(r"[a-z0-9]+", text.lower())


# Build (Once) a BM25 Index Over Every Chunk in the Vector Store
def _get_bm25():
    global _bm25, _bm25_chunks
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        _bm25_chunks = vectorstore.all_chunks()
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _bm25_chunks])
    return _bm25, _bm25_chunks


# Top-n Chunks by BM25 Keyword Score
def _bm25_candidates(query: str, n: int) -> list[dict]:
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(_tokenize(query))
    n = min(n, len(scores))
    # argpartition Finds the Top-n in O(N) Without Sorting the Rest, Then We Sort
    # Just Those n by Score Descending - Cheaper Than a Full Sort on a Big Corpus
    top = np.argpartition(scores, -n)[-n:]
    top = top[np.argsort(scores[top])[::-1]]
    return [chunks[i] for i in top]


# Fuse Vector + BM25 Rankings With RRF, Return the Top-n Fused Chunks
def hybrid(query: str, query_vector: list[float], n: int, rrf_k: int = config.RRF_K) -> list[dict]:
    vec_hits = vectorstore.search(query_vector, k=n)
    bm25_hits = _bm25_candidates(query, n)

    # RRF: Each List Contributes 1/(rrf_k + rank) to a Chunk, Rank Starting at 1
    # A Chunk Both Lists Rank Highly Bubbles Up; rrf_k Damps Top-Rank Dominance
    fused: dict[str, float] = {}
    info: dict[str, dict] = {}
    for rank, h in enumerate(vec_hits, start=1):
        fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (rrf_k + rank)
        info[h["id"]] = h  # Vector Hit Carries Similarity as "score"
    for rank, h in enumerate(bm25_hits, start=1):
        fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (rrf_k + rank)
        info.setdefault(h["id"], {**h, "score": None})  # BM25-Only Chunk Has No Similarity

    out = []
    for cid in sorted(fused, key=lambda c: fused[c], reverse=True)[:n]:
        h = dict(info[cid])
        h["fusion_score"] = round(fused[cid], 4)
        out.append(h)
    return out


# Drop the Cached BM25 Index; Call After Re-Indexing So It Rebuilds
def reset_cache() -> None:
    global _bm25, _bm25_chunks
    _bm25 = _bm25_chunks = None
