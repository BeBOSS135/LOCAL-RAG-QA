# Evaluation - Retrieval Metrics + LLM-Judged Generation Metrics
# Run:  python evaluate.py            # Retrieval Comparison
#       python evaluate.py --judge    # + Faithfulness/Relevancy (Slow, Local LLM)
import json
import re
import statistics
import sys
import time

import config
import embeddings
import reranker
import retrieval
import vectorstore

EVAL_SET = json.loads((config.PROJECT_ROOT / "eval_set.json").read_text(encoding="utf-8"))

# Retrieval Configurations to Compare
CONFIGS = {
    "vector":         dict(use_hybrid=False, use_rerank=False),
    "vector+rerank":  dict(use_hybrid=False, use_rerank=True),
    "hybrid+rerank":  dict(use_hybrid=True,  use_rerank=True),
}


# Return the Ordered Source Filenames for a Question (No Generation)
def _retrieve_sources(question, k, use_hybrid, use_rerank):
    q_vec = embeddings.embed_one(question)
    n = config.RETRIEVE_N if (use_hybrid or use_rerank) else k
    hits = retrieval.hybrid(question, q_vec, n) if use_hybrid else vectorstore.search(q_vec, k=n)
    if use_rerank:
        hits = reranker.rerank(question, hits, top_k=k)
    else:
        hits = hits[:k]
    return [h["source"] for h in hits]


# 1-Based Rank of the First Source Matching Any Expected Lecture Tag, Else 0
def _first_hit_rank(sources, expected):
    for rank, src in enumerate(sources, start=1):
        if any(tag in src for tag in expected):
            return rank
    return 0


def retrieval_eval(k=config.TOP_K):
    print(f"\nRetrieval eval  ({len(EVAL_SET)} questions, k={k})")
    print(f"{'config':<16}{'hit-rate@k':>12}{'MRR':>8}{'latency(ms)':>14}")
    for name, cfg in CONFIGS.items():
        hits, rrs, lats = 0, [], []
        for item in EVAL_SET:
            t0 = time.perf_counter()
            sources = _retrieve_sources(item["question"], k, **cfg)
            lats.append((time.perf_counter() - t0) * 1000)
            rank = _first_hit_rank(sources, item["expected"])
            hits += rank > 0
            rrs.append(1.0 / rank if rank else 0.0)
        print(f"{name:<16}{hits / len(EVAL_SET):>12.2f}"
              f"{statistics.mean(rrs):>8.2f}{statistics.median(lats):>14.0f}")


FAITHFULNESS_PROMPT = (
    "You are grading a RAG answer. Given the CONTEXT and the ANSWER, score how "
    "fully the answer's factual claims are supported by the context: 1.0 = every "
    "claim is supported, 0.0 = claims are unsupported or contradicted by the context.\n"
    'Respond with ONLY JSON: {{"score": <0.0-1.0>, "reason": "<short>"}}\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)

RELEVANCY_PROMPT = (
    "You are grading a RAG answer. Given the QUESTION and the ANSWER, score how "
    "directly the answer addresses the question: 1.0 = fully on-point, 0.0 = "
    "irrelevant or evasive.\n"
    'Respond with ONLY JSON: {{"score": <0.0-1.0>, "reason": "<short>"}}\n\n'
    "QUESTION:\n{question}\n\nANSWER:\n{answer}"
)


# Ask the Local LLM for a 0-1 Score; Parse the JSON It Returns
def _judge(client, prompt: str) -> float:
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},  # Deterministic Grading
    )
    text = resp["message"]["content"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return max(0.0, min(1.0, float(json.loads(match.group(0))["score"])))
    except Exception:
        return float("nan")


# Faithfulness + Answer Relevancy, Scored by the Local LLM as Judge
def judge_eval(k=config.TOP_K):
    import ollama

    import rag

    client = ollama.Client(host=config.OLLAMA_HOST)
    print(f"\nLLM-judge eval  ({len(EVAL_SET)} questions, local judge - this is slow)")

    faiths, relevs = [], []
    for item in EVAL_SET:
        res = rag.answer(item["question"], k=k)
        context = "\n\n".join(s["text"] for s in res["sources"])
        f = _judge(client, FAITHFULNESS_PROMPT.format(context=context, answer=res["answer"]))
        r = _judge(client, RELEVANCY_PROMPT.format(question=item["question"], answer=res["answer"]))
        faiths.append(f)
        relevs.append(r)
        print(f"  faith={f:.2f} relev={r:.2f}  {item['question'][:45]}")

    clean = lambda xs: [x for x in xs if x == x]  # Drop NaNs From Parse Failures
    print(f"\nMean faithfulness: {statistics.mean(clean(faiths)):.3f}  (target > 0.85)")
    print(f"Mean relevancy:    {statistics.mean(clean(relevs)):.3f}")


if __name__ == "__main__":
    retrieval_eval()
    if "--judge" in sys.argv:
        judge_eval()
