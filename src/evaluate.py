# Evaluation - Retrieval Metrics + Abstention Gate + LLM-Judged Generation Metrics
# Run:  python evaluate.py            # Retrieval Comparison (Answerable Set)
#       python evaluate.py --abstain  # + Abstention Gate: false-refusal vs correct-abstention
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

# Prefer a Local, Gitignored Eval Set (Your Own Vault Questions) if Present; Otherwise
# the Generic Sample Eval Shipped With the Repo (Runs Out-of-Box on data/sample.md)
_eval_path = config.PROJECT_ROOT / "eval_set.local.json"
if not _eval_path.exists():
    _eval_path = config.PROJECT_ROOT / "eval_set.json"
EVAL_SET = json.loads(_eval_path.read_text(encoding="utf-8"))

# Answerable Items Carry "expected" Source Tags; the Unanswerable Slice ("answerable": false)
# Has None and Is Used Only by the Abstention Eval, So Retrieval/Judge Evals Skip It
ANSWERABLE = [it for it in EVAL_SET if it.get("answerable", True)]
UNANSWERABLE = [it for it in EVAL_SET if not it.get("answerable", True)]

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
    return [h["source_path"] for h in hits]


# 1-Based Rank of the First Source Matching Any Expected Lecture Tag, Else 0
def _first_hit_rank(sources, expected):
    for rank, src in enumerate(sources, start=1):
        if any(tag in src for tag in expected):
            return rank
    return 0


def retrieval_eval(k=config.TOP_K):
    print(f"\nRetrieval eval  ({len(ANSWERABLE)} answerable questions, k={k})")
    print(f"{'config':<16}{'hit-rate@k':>12}{'MRR':>8}{'latency(ms)':>14}")
    for name, cfg in CONFIGS.items():
        hits, rrs, lats = 0, [], []
        for item in ANSWERABLE:
            t0 = time.perf_counter()
            sources = _retrieve_sources(item["question"], k, **cfg)
            lats.append((time.perf_counter() - t0) * 1000)
            rank = _first_hit_rank(sources, item["expected"])
            hits += rank > 0
            rrs.append(1.0 / rank if rank else 0.0)
        print(f"{name:<16}{hits / len(ANSWERABLE):>12.2f}"
              f"{statistics.mean(rrs):>8.2f}{statistics.median(lats):>14.0f}")


# Abstention Eval - Does the Gate Refuse the Out-of-Corpus Questions Without Wrongly
# Refusing the Answerable Ones? Prints Each Top Score So the Threshold Can Be Tuned to
# the Gap Between the Two Groups (Run With the Default Pipeline: Hybrid + Re-Rank)
def abstain_eval(k=config.TOP_K):
    import rag

    print(f"\nAbstention eval  (gate {'ON' if config.USE_ABSTAIN else 'OFF'}, "
          f"rerank floor {config.ABSTAIN_MIN_RERANK}, similarity floor {config.ABSTAIN_MIN_SIMILARITY})")

    def run(items, label):
        rows = []
        for it in items:
            r = rag.retrieve(it["question"], k=k)
            rows.append((r["abstain"],))
            verdict = "REFUSE" if r["abstain"] else "answer"
            print(f"  {label:<12} rerank={str(r['top_score']):<6} sim={str(r['top_sim']):<6} "
                  f"{verdict:<7} {it['question'][:46]}")
        return rows

    a = run(ANSWERABLE, "answerable")
    u = run(UNANSWERABLE, "unanswerable")
    false_refuse = sum(1 for (ab,) in a if ab)
    correct_abstain = sum(1 for (ab,) in u if ab)
    print(f"\nFalse-refusal rate (answerable):    {false_refuse}/{len(a)} = {false_refuse / len(a):.2f}  (want 0.00)")
    print(f"Correct-abstention rate (out-corpus): {correct_abstain}/{len(u)} = {correct_abstain / len(u):.2f}  (want 1.00)")


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
    print(f"\nLLM-judge eval  ({len(ANSWERABLE)} answerable questions, local judge - this is slow)")

    faiths, relevs = [], []
    for item in ANSWERABLE:
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
    if "--abstain" in sys.argv:
        abstain_eval()
    if "--judge" in sys.argv:
        judge_eval()
