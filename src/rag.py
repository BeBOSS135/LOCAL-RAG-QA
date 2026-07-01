# Query Orchestration: Rewrite -> Embed -> Retrieve -> Union -> Re-Rank -> Dedup -> Gate -> Generate -> Verify -> Cite
import re
import time

import config
import embeddings
import llm
import reranker
import retrieval
import rewrite
import vectorstore


def _search(query: str, use_hybrid: bool, n: int,
            where: dict | None = None, scope_prefix: str | None = None) -> list[dict]:
    q_vec = embeddings.embed_one(query)
    return (retrieval.hybrid(query, q_vec, n, where=where, scope_prefix=scope_prefix)
            if use_hybrid else vectorstore.search(q_vec, k=n, where=where))


# Human-Readable Citation for a Hit: "Title › Heading" (md) / "file › p.3" (pdf)
# / Just the Title When Flat. The Reference Point the User Sees and the LLM Cites
def _ref(h: dict) -> str:
    title = h.get("title") or h.get("source_path") or "?"
    # Drop Heading Levels That Just Repeat the Title (Avoids "Guide › Guide")
    parts = [p for p in (h.get("heading_path") or "").split(" > ") if p and p != title]
    return f"{title} › " + " › ".join(parts) if parts else title


# Token Set of a Chunk's Text - Basis for Cheap Near-Duplicate Detection (Jaccard)
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# Drop Hits Whose Text Nearly Duplicates a Higher-Ranked Hit (Token-Set Jaccard >=
# Floor), Keeping the First (Best-Ranked) of Each Duplicate Group. O(n^2) but n Is the
# Small Ranked Candidate Pool, Not the Corpus - So It's Scale-Safe
def _dedup(hits: list[dict]) -> list[dict]:
    kept: list[dict] = []
    seen: list[set[str]] = []
    for h in hits:
        toks = _tokens(h["text"])
        if any(toks and s and len(toks & s) / len(toks | s) >= config.DEDUP_JACCARD for s in seen):
            continue
        kept.append(h)
        seen.append(toks)
    return kept


# Abstention Decision - Is the Best Hit Strong Enough to Trust? Two Signals: the Top
# Re-Rank Score (hits Are Best-First) and the Highest Cosine Similarity Among Hits.
# Answer if EITHER Clears Its Floor; Abstain Only When BOTH Are Weak - the Cross-Encoder
# Collapses on Terse Queries Where Cosine Still Separates In- vs Out-of-Corpus (See config).
# Returns (abstain, rerank_top, sim_top, signal)
def _abstain(hits: list[dict], use_rerank: bool) -> tuple[bool, float | None, float | None, str]:
    if not config.USE_ABSTAIN or not hits:
        return (not hits, None, None, "none")
    # Max Cosine Over Hits (BM25-Only Chunks Have score=None, So Skip Them)
    sims = [h["score"] for h in hits if h.get("score") is not None]
    sim_top = round(max(sims), 3) if sims else 0.0
    sim_ok = sim_top >= config.ABSTAIN_MIN_SIMILARITY
    if use_rerank and hits[0].get("rerank_score") is not None:
        rr_top = round(hits[0]["rerank_score"], 3)
        return (not (rr_top >= config.ABSTAIN_MIN_RERANK or sim_ok), rr_top, sim_top, "rerank|sim")
    return (not sim_ok, None, sim_top, "sim")


# Everything Up to Generation - Returns the Chosen Chunks + Context Blocks
def retrieve(
    question: str,
    k: int = config.TOP_K,
    use_rerank: bool = config.USE_RERANK,
    use_hybrid: bool = config.USE_HYBRID,
    use_rewrite: bool = config.USE_REWRITE,
    retrieve_n: int = config.RETRIEVE_N,
    scope: str | None = None,
    tag: str | None = None,
) -> dict:
    t0 = time.perf_counter()

    # Optional Scope: Restrict Retrieval to a Source Root / Folder Before ANN. where
    # Pre-Filters the Vector Side (Chroma); scope_prefix Filters the BM25 Side to Match
    where = vectorstore.scope_where(scope)
    scope_prefix = scope or None

    # Expand Into Alternate Phrasings So Source Wording Need Not Match the User's
    rewrites = rewrite.expand(question) if use_rewrite else []
    queries = [question] + rewrites

    # Over-Fetch per Phrasing When Re-Ranking, Then Union (Dedupe by Chunk id).
    # A Tag Filter Is a Post-Filter, So Fetch a Bigger Pool to Keep Recall (See Config)
    n = retrieve_n if (use_rerank or use_hybrid or use_rewrite) else k
    if tag:
        n = max(n, config.TAG_FILTER_FETCH)
    by_id: dict[str, dict] = {}
    for q in queries:
        for h in _search(q, use_hybrid, n, where=where, scope_prefix=scope_prefix):
            by_id.setdefault(h["id"], h)  # First Occurrence Wins
    candidates = list(by_id.values())

    # Tag Post-Filter: Keep Only Chunks Carrying the Tag (Stored Delimited "|a|b|",
    # So Match the Whole Bar-Wrapped Token - No Prefix False Positives)
    if tag:
        candidates = [c for c in candidates if f"|{tag}|" in c.get("tags", "")]

    # Rank ALL Candidates Against the ORIGINAL Question (Not the Rewrites), Then Drop
    # Near-Duplicates, Then Keep Top-K - So Repeated Content Doesn't Eat Top-K Slots
    if use_rerank:
        ranked = reranker.rerank(question, candidates, top_k=len(candidates))
    else:
        score = lambda h: h.get("fusion_score") or h.get("score") or 0
        ranked = sorted(candidates, key=score, reverse=True)
    hits = (_dedup(ranked) if config.DEDUP else ranked)[:k]

    # Weak Retrieval -> Abstain So Callers Can Refuse Rather Than Hallucinate
    abstain, top_score, top_sim, signal = _abstain(hits, use_rerank)

    return {
        "question": question,
        "scope": scope,
        "tag": tag,
        "rewrites": rewrites,
        "abstain": abstain,
        "top_score": top_score,
        "top_sim": top_sim,
        "abstain_signal": signal,
        "blocks": [f"[{_ref(h)}] {h['text']}" for h in hits],
        "sources": [
            {
                "source": _ref(h),  # Human Reference: "Title › Heading" / "file › p.3"
                "source_path": h.get("source_path"),
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


# Post-Generation Grounding Check, Gated to the GRAY ZONE (Skipped When the Top Rerank
# Score Is Already High, to Save an LLM Call). Returns (supported, reason); supported Is
# None When Not Checked. Non-Destructive - the Caller Decides How to Surface a Failure
def verify_answer(result: dict, answer: str) -> tuple[bool | None, str]:
    top = result.get("top_score")
    if (not config.VERIFY_ANSWERS or result.get("abstain")
            or (top is not None and top >= config.VERIFY_SKIP_ABOVE)):
        return (None, "")
    return llm.verify(result["blocks"], answer)


# Full Pipeline Including Generation (Non-Streaming) - Used by CLI + Eval
# On Abstain, Return the Fixed Refusal and Skip the LLM Entirely (Faster, No Hallucination).
# Otherwise Generate, Then Grounding-Check in the Gray Zone and Caveat an Unsupported Answer
def answer(question: str, **kwargs) -> dict:
    t0 = time.perf_counter()
    result = retrieve(question, **kwargs)
    if result["abstain"]:
        result["answer"] = llm.REFUSAL
        result["verified"] = None
    else:
        ans = llm.generate(question, result["blocks"])
        supported, reason = verify_answer(result, ans)
        result["verified"] = supported
        result["verify_reason"] = reason
        result["answer"] = (llm.UNVERIFIED_CAVEAT + ans) if supported is False else ans
    result["latency_s"] = round(time.perf_counter() - t0, 2)
    return result
