# Lightweight Usage + Feedback Logging to JSONL. Every Query and Every Thumbs Rating
# Is Appended to logs/ - Over Time This Grows a REAL Eval Set From Actual Use (Better
# Than a Synthetic One) and Enables Retrieval-Drift Monitoring / Hard-Negative Mining
# Later. Deliberately Just Append-Only JSONL - No DB, No Online Learning (Over-Build).
# logs/ Is gitignored: These Records Are the User's Own Queries (Personal), Never Pushed.
import json
import time
from pathlib import Path

import config

LOG_DIR = config.PROJECT_ROOT / "logs"
QUERY_LOG = LOG_DIR / "queries.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"


def _append(path: Path, record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# One Line per Query: What Was Asked, the Scope/Tag, Whether It Abstained, the Gate
# Scores, and Which Sources Came Back - Enough to Later Curate Answerable/Unanswerable
# Eval Slices and Spot Retrieval Drift Without Re-Running Anything
def log_query(question: str, result: dict) -> None:
    _append(QUERY_LOG, {
        "question": question,
        "scope": result.get("scope"),
        "tag": result.get("tag"),
        "abstain": result.get("abstain"),
        "top_score": result.get("top_score"),
        "top_sim": result.get("top_sim"),
        "verified": result.get("verified"),
        "sources": [s["source_path"] for s in result.get("sources", [])],
    })


# One Line per Thumbs Rating ("up"/"down") Tied to the Question + Answer It Judged -
# The Human Signal for Later Reranker Tuning / Eval Curation
def log_feedback(question: str, answer: str, rating: str) -> None:
    _append(FEEDBACK_LOG, {"question": question, "answer": answer, "rating": rating})
