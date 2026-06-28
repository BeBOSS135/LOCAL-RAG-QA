"""Central config for the RAG pipeline. One place to tune every knob."""
# Pre-load pyarrow before torch/sklearn ever load. Deep in sentence-transformers'
# import chain, sklearn->pandas triggers pyarrow's native lib, which segfaults on
# Windows when torch is already loaded. Importing it first (config is imported
# before torch everywhere) sidesteps the DLL clash. No-op if pyarrow is absent.
try:
    import pyarrow  # noqa: F401
except Exception:
    pass

from pathlib import Path

# Paths. Code lives in src/; data, index and eval_set sit at the repo root one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # drop documents to index here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # persisted vector store

# Chunking (token-based). Capped at the embedding model's max sequence length so
# no chunk is silently truncated when embedded — word counts can't guarantee this.
CHUNK_TOKENS = 400          # bge-base allows 512; 400 keeps room for special tokens
CHUNK_OVERLAP_TOKENS = 60   # ~15% overlap to preserve context across boundaries

# OCR fallback — slide/diagram PDFs carry content in images the text layer can't
# see. We OCR a page only when its extracted text is sparse, so text-rich PDFs stay
# fast while image-heavy slides still get read. Dataset-agnostic by design.
USE_OCR = True
OCR_MIN_CHARS = 120  # text-layer length below which a page is OCR'd instead
OCR_DPI = 200        # render resolution for OCR (higher = slower, more accurate)
OCR_LANGS = ["en"]

# Embeddings — bge-base-en-v1.5 (768-dim) ranks far better than MiniLM on retrieval
# and still runs fast on a 4060. bge wants queries (not passages) prefixed with an
# instruction; QUERY_PREFIX is applied only to query embeddings (see embeddings.py).
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_DEVICE = "cuda"             # auto-falls back to cpu in embeddings.py
EMBED_BATCH = 128                 # larger batch = better GPU utilisation
NORMALIZE = True                  # unit vectors so cosine == dot product, stable

# Vector store
COLLECTION_NAME = "documents"
ADD_BATCH = 5000  # Chroma rejects huge single adds; chunk the insert for big datasets
# HNSW index params — higher = better recall at the cost of build time / memory.
# These are dataset-size dependent, not data-specific, so safe general defaults.
HNSW_M = 32
HNSW_CONSTRUCTION_EF = 200
HNSW_SEARCH_EF = 100

# Retrieval
RETRIEVE_N = 10  # initial vector candidates pulled before re-ranking
TOP_K = 4        # chunks kept (after re-rank) and fed to the LLM

# Query rewriting — expand the question into alternate phrasings (+ a HyDE answer
# passage) before retrieval, so a query's wording doesn't have to match the source's.
# Off by default: it costs two extra LLM calls (~doubles latency) and only helps
# ordinary vocabulary gaps, so reach for it (UI toggle) when a query returns weak
# sources rather than paying the cost on every question.
USE_REWRITE = False
REWRITE_N = 3  # number of alternate phrasings generated per question

# Hybrid retrieval (BM25 keyword + vector, fused with Reciprocal Rank Fusion)
USE_HYBRID = True
RRF_K = 60  # RRF smoothing constant; larger = flatter rank weighting

# Re-ranking — bge-reranker-base is a stronger cross-encoder than ms-marco-MiniLM,
# giving sharper final ordering (it reorders RETRIEVE_N candidates, keeps TOP_K).
USE_RERANK = True
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_BATCH = 64  # scores all candidates in batches; matters if RETRIEVE_N grows

# LLM (Ollama, local)
OLLAMA_MODEL = "mistral"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = "30m"  # keep the model resident in VRAM between queries

