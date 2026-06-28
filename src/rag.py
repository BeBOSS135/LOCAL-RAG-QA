"""Query phase orchestration:
rewrite -> embed -> retrieve (hybrid) per phrasing -> union -> re-rank -> generate -> cite.

`retrieve()` does everything up to (not including) generation, so callers can show
sources immediately and stream the answer. `answer()` is the non-streaming convenience
wrapper used by the CLI and the evaluation suite.
"""
import time

import config
import embeddings
import llm
import reranker
import retrieval
import rewrite
import vectorstore


def _search(query: str, use_hybrid: bool, n: int) -> list[dict]:
    q_vec = embeddings.embed_one(query)
    return retrieval.hybrid(query, q_vec, n) if use_hybrid else vectorstore.search(q_vec, k=n)


def retrieve(
    question: str,
    k: int = config.TOP_K,
    use_rerank: bool = config.USE_RERANK,
    use_hybrid: bool = config.USE_HYBRID,
    use_rewrite: bool = config.USE_REWRITE,
    retrieve_n: int = config.RETRIEVE_N,
) -> dict:
    """Everything up to generation: returns the chosen chunks + context blocks."""
    t0 = time.perf_counter()

    # Expand into alternate phrasings so source wording need not match the user's
    rewrites = rewrite.expand(question) if use_rewrite else []
    queries = [question] + rewrites

    # Over-fetch per phrasing when re-ranking, then union (dedupe by chunk id)
    n = retrieve_n if (use_rerank or use_hybrid or use_rewrite) else k
    by_id: dict[str, dict] = {}
    for q in queries:
        for h in _search(q, use_hybrid, n):
            by_id.setdefault(h["id"], h)  # first occurrence wins
    candidates = list(by_id.values())

    # Always judge relevance against the ORIGINAL question, not the rewrites
    if use_rerank:
        hits = reranker.rerank(question, candidates, top_k=k)
    else:
        score = lambda h: h.get("fusion_score") or h.get("score") or 0
        hits = sorted(candidates, key=score, reverse=True)[:k]

    return {
        "question": question,
        "rewrites": rewrites,
        "blocks": [f"[{h['source']}] {h['text']}" for h in hits],
        "sources": [
            {
                "source": h["source"],
                "score": round(h["score"], 3) if h.get("score") is not None else None,
                "fusion_score": h.get("fusion_score"),
                "rerank_score": round(h["rerank_score"], 3) if "rerank_score" in h else None,
                "text": h["text"],
            }
            for h in hits
        ],
        "retrieval": "hybrid" if use_hybrid else "vector",
        "reranked": use_rerank,
        "retrieve_s": round(time.perf_counter() - t0, 2),
    }


def answer(question: str, **kwargs) -> dict:
    """Full pipeline including generation (non-streaming). Used by CLI + eval."""
    t0 = time.perf_counter()
    result = retrieve(question, **kwargs)
    result["answer"] = llm.generate(question, result["blocks"])
    result["latency_s"] = round(time.perf_counter() - t0, 2)
    return result
