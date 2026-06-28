"""Cross-encoder re-ranking. Vector search is fast but coarse; the cross-encoder
reads (query, chunk) together for a sharper relevance score, then we keep top_k."""
import config

_model = None  # lazy singleton, loads once per process


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder
        device = config.EMBED_DEVICE if torch.cuda.is_available() else "cpu"
        print(f"Loading reranker '{config.RERANK_MODEL}' on {device}")
        _model = CrossEncoder(config.RERANK_MODEL, device=device)
    return _model


def rerank(query: str, hits: list[dict], top_k: int = config.TOP_K) -> list[dict]:
    """Re-score candidate hits against the query and return the top_k."""
    if not hits:
        return hits
    model = _get_model()
    scores = model.predict([(query, h["text"]) for h in hits], batch_size=config.RERANK_BATCH)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
