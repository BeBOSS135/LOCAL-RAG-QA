"""CLI: ask a question against the indexed documents (streams the answer)."""
import sys
import time

import llm
from rag import retrieve


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
    for token in llm.generate_stream(question, r["blocks"]):  # stream as it generates
        print(token, end="", flush=True)
    print()

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
