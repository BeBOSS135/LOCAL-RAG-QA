"""Embedding model wrapper. Same model is used to index and to query."""
import config

# Silence the expected "sequence longer than 256" notice — we tokenize whole docs
# only to get offsets for chunking, never to run them through the model
from transformers.utils import logging as _hf_logging
_hf_logging.set_verbosity_error()

_model = None      # lazy singletons, loaded once per process
_tokenizer = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        # Honor configured device but degrade gracefully without CUDA
        device = config.EMBED_DEVICE if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{config.EMBED_MODEL}' on {device}")
        _model = SentenceTransformer(config.EMBED_MODEL, device=device)
    return _model


def get_tokenizer():
    """The embedding model's tokenizer (fast), for token-aware chunking.

    Loaded via AutoTokenizer — NOT from the SentenceTransformer — so chunking never
    pulls the GPU model into the process. That keeps the OCR/extraction pass (which
    chunks) free of the embedder, letting it run in a separate process: EasyOCR and
    the SentenceTransformer segfault if they share one. The bare model name 401s on
    the Hub, so we add the canonical `sentence-transformers/` prefix.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        repo = config.EMBED_MODEL
        if "/" not in repo:
            repo = f"sentence-transformers/{repo}"
        _tokenizer = AutoTokenizer.from_pretrained(repo)
    return _tokenizer


def embed(texts: list[str]) -> list[list[float]]:
    """Encode a batch of passages into vectors (no query prefix — index side)."""
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=config.NORMALIZE,  # unit length -> stable cosine
        show_progress_bar=len(texts) > 256,
    )
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    """Encode a single query. bge wants queries (not passages) instruction-prefixed,
    so the asymmetry lives here — every query path goes through embed_one."""
    return embed([config.QUERY_PREFIX + text])[0]
