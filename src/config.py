# Config

# Pre-Import Pyarrow Before Torch to Dodge a Windows DLL Segfault
# sklearn->pandas Loads Pyarrow's Native Lib, Which Crashes if Torch Loaded First
# Config Is Imported Before Torch Everywhere, So Importing Here Wins the Race
try:
    import pyarrow  # noqa: F401
except Exception:
    pass

from pathlib import Path

# Paths - Code in src/, Data and Index One Level Up at Repo Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # Drop Documents to Index Here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # Persisted Vector Store

# Token-Based Chunking, Capped at the Model's Max Sequence Length
# So No Chunk Is Silently Truncated When Embedded - Word Counts Can't Guarantee This
CHUNK_TOKENS = 400          # bge-base Allows 512; 400 Leaves Room for Special Tokens
CHUNK_OVERLAP_TOKENS = 60   # ~15% Overlap to Preserve Context Across Boundaries

# OCR Fallback for Slide/Diagram PDFs That Hide Content in Images
# OCR a Page Only When Its Text Is Sparse, So Text-Rich PDFs Stay Fast
USE_OCR = True
OCR_MIN_CHARS = 120  # Text-Layer Length Below Which a Page Is OCR'd
OCR_DPI = 200        # Render Resolution (Higher = Slower, More Accurate)
OCR_LANGS = ["en"]

# Embeddings - bge-base-en-v1.5 (768-dim) Beats MiniLM on Retrieval, Still Fast on a 4060
# bge Wants Queries (Not Passages) Instruction-Prefixed; Prefix Applied Only to Queries
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_DEVICE = "cuda"             # Auto-Falls Back to CPU in embeddings.py
EMBED_BATCH = 128                 # Larger Batch = Better GPU Utilisation
NORMALIZE = True                  # Unit Vectors So Cosine == Dot Product

# Vector Store
COLLECTION_NAME = "documents"
ADD_BATCH = 5000  # Chroma Rejects Huge Single Adds, So Batch the Insert
# HNSW Params - Higher = Better Recall at the Cost of Build Time / Memory
# Dataset-Size Dependent, Not Data-Specific, So Safe as General Defaults
HNSW_M = 32
HNSW_CONSTRUCTION_EF = 200
HNSW_SEARCH_EF = 100

# Retrieval
RETRIEVE_N = 10  # Initial Candidates Pulled Before Re-Ranking
TOP_K = 4        # Chunks Kept After Re-Rank and Fed to the LLM

# Query Rewriting - Expand Into Alternate Phrasings (+ HyDE Answer) Before Retrieval
# So Query Wording Need Not Match the Source's
# Off by Default: Two Extra LLM Calls (~Doubles Latency); Toggle On for Weak Sources
USE_REWRITE = False
REWRITE_N = 3  # Alternate Phrasings Generated per Question

# Hybrid Retrieval - BM25 Keyword + Vector, Fused With RRF
USE_HYBRID = True
RRF_K = 60  # RRF Smoothing Constant; Larger = Flatter Rank Weighting

# Re-Ranking - bge-reranker-base Is a Stronger Cross-Encoder Than ms-marco-MiniLM
# Reorders RETRIEVE_N Candidates Down to TOP_K
USE_RERANK = True
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_BATCH = 64  # Scores Candidates in Batches; Matters if RETRIEVE_N Grows

# LLM - Ollama, Local
OLLAMA_MODEL = "mistral"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = "30m"  # Keep Model Resident in VRAM Between Queries
