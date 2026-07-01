# Embeddings
import config

# Silence the Expected "Sequence Longer Than 256" Notice - We Tokenize Whole Docs
# Only to Get Offsets for Chunking, Never to Run Them Through the Model
from transformers.utils import logging as _hf_logging
_hf_logging.set_verbosity_error()

_model = None      # Lazy Singletons, Loaded Once per Process
_tokenizer = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        # Honor Configured Device but Degrade Gracefully Without CUDA
        device = config.EMBED_DEVICE if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{config.EMBED_MODEL}' on {device}")
        _model = SentenceTransformer(config.EMBED_MODEL, device=device, revision=config.EMBED_REVISION)
    return _model


# Tokenizer Loaded via AutoTokenizer, NOT the SentenceTransformer, So Chunking
# Never Pulls the GPU Model Into This Process - Lets Extraction/OCR Run Separately
# (EasyOCR and the Embedder Segfault if They Share One Process)
# Bare Model Name 401s on the Hub, So Add the Canonical sentence-transformers/ Prefix
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        repo = config.EMBED_MODEL
        if "/" not in repo:
            repo = f"sentence-transformers/{repo}"
        _tokenizer = AutoTokenizer.from_pretrained(repo, revision=config.EMBED_REVISION)
    return _tokenizer


# Encode a Batch of Passages (No Query Prefix - Index Side)
def embed(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=config.NORMALIZE,  # Unit Length -> Stable Cosine
        show_progress_bar=len(texts) > 256,
    )
    return vectors.tolist()


# Encode a Single Query - bge Wants Queries Instruction-Prefixed, So the
# Asymmetry Lives Here; Every Query Path Goes Through embed_one
def embed_one(text: str) -> list[float]:
    return embed([config.QUERY_PREFIX + text])[0]
