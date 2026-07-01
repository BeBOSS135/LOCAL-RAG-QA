# Query CLI
import sys
import time

import feedback
import llm
from rag import retrieve, verify_answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        return
    question = " ".join(sys.argv[1:])

    t0 = time.perf_counter()
    r = retrieve(question)

    print(f"\nQ: {question}")
    if r["rewrites"]:
        print("  (also searched: " + " | ".join(r["rewrites"]) + ")")

    print("\nA: ", end="", flush=True)
    if r["abstain"]:
        # Honor the Gate Here Too - Weak Retrieval Refuses Instead of Generating
        answer_text, verified = llm.REFUSAL, None
        print(answer_text)
    else:
        parts = []
        for token in llm.generate_stream(question, r["blocks"]):  # Stream as It Generates
            print(token, end="", flush=True)
            parts.append(token)
        print()
        answer_text = "".join(parts)
        verified, reason = verify_answer(r, answer_text)  # Gray-Zone Grounding Check
        if verified is False:
            print(f"\n⚠️  {llm.UNVERIFIED_CAVEAT.strip()}  ({reason})")

    feedback.log_query(question, {**r, "verified": verified})

    tag = f"{r['retrieval']}{', re-ranked' if r['reranked'] else ''}"
    print(f"\nSources ({round(time.perf_counter() - t0, 2)}s, {tag}):")
    for s in r["sources"]:
        parts = []
        if s["score"] is not None:
            parts.append(f"sim {s['score']}")
        if s["fusion_score"] is not None:
            parts.append(f"fusion {s['fusion_score']}")
        if s["rerank_score"] is not None:
            parts.append(f"rerank {s['rerank_score']}")
        print(f"  - {s['source']} ({', '.join(parts)})")


if __name__ == "__main__":
    main()
