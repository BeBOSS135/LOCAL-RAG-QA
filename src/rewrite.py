"""Query rewriting to bridge vocabulary gaps between question and source.

Retrieval is vocabulary-sensitive — a chunk ranks high only if its wording matches
the query's. Two complementary expansions, retrieved alongside the original question:

  - Rephrasings: synonym/keyword variety (helps exact-term and BM25 matching).
  - HyDE (Hypothetical Document Embeddings): a generated *answer* passage. Source
    text is usually phrased like an explanation, not a question, so embedding a
    plausible answer matches it far better — this is what catches a lecture that
    describes "what percentage to remember" when you asked about "LSTM gates".

The HyDE passage may be wrong; that's fine. It's only used to FIND chunks. The final
answer is generated from the real retrieved text and re-ranked against the original
question, so a hallucinated hypothesis can't leak into the output.
"""
import ollama

import config

REPHRASE_PROMPT = (
    "Generate {n} alternative search queries for the question below, using different "
    "wording and synonyms. Output ONLY the queries, one per line, no numbering.\n\n"
    "Question: {q}"
)

HYDE_PROMPT = (
    "Write a short, factual 2-3 sentence passage that answers the question, as if it "
    "were an excerpt from lecture notes — explain the concept in plain, descriptive "
    "language. Output only the passage.\n\nQuestion: {q}"
)


def _chat(prompt: str, temperature: float) -> str:
    client = ollama.Client(host=config.OLLAMA_HOST)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return resp["message"]["content"].strip()


def expand(question: str, n: int = config.REWRITE_N) -> list[str]:
    """Alternate retrieval queries: n keyword rephrasings + one HyDE answer passage.

    Best-effort — returns whatever it can; retrieval still works if this yields nothing.
    """
    extra: list[str] = []
    try:
        lines = [l.strip(" -*\t").strip() for l in _chat(REPHRASE_PROMPT.format(n=n, q=question), 0.3).splitlines()]
        seen = {question.lower()}
        for l in lines:
            if l and l.lower() not in seen:
                seen.add(l.lower())
                extra.append(l)
        extra = extra[:n]
    except Exception:
        pass
    try:
        hyde = _chat(HYDE_PROMPT.format(q=question), 0.2)
        if hyde:
            extra.append(hyde)
    except Exception:
        pass
    return extra
