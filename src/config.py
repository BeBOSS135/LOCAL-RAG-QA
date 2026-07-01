# Config

# Pre-Import Pyarrow Before Torch to Dodge a Windows DLL Segfault
# sklearn->pandas Loads Pyarrow's Native Lib, Which Crashes if Torch Loaded First
# Config Is Imported Before Torch Everywhere, So Importing Here Wins the Race
try:
    import pyarrow  # noqa: F401
except Exception as _e:
    # Don't Fail Startup - but Surface It. If This Silently Skips, a Later torch/pyarrow
    # DLL Segfault (No Traceback on Windows) Is Baffling to Debug
    import sys
    print(f"config: pyarrow pre-import skipped ({_e}) - watch for DLL segfaults", file=sys.stderr)

import json
from pathlib import Path

# Paths - Code in src/, Data and Index One Level Up at Repo Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"          # Drop Documents to Index Here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"   # Persisted Vector Store

# Sources to Index - Each Root Gets a Short Name; a Chunk's source_path Is
# "<name>/<relpath>", So Files Are Globally Unique Across Roots (No Filename
# Collisions at Scale). The Loader Registry Handles Whatever File Types Show Up.
#
# Personal Roots (e.g. Your Obsidian Vault) Do NOT Belong Here - Keep Real Machine
# Paths Out of Version Control. Instead Drop a Gitignored `sources.local.json` at the
# Repo Root; It's Merged in Below. Example:
#   [{"name": "vault", "path": "C:/path/to/your/vault", "exclude": ["Archive"]}]
SOURCES = [
    {"name": "data", "path": DATA_DIR},
]

# Merge in Local, Gitignored Source Roots So Personal Paths Never Touch Tracked Files
_local_sources = PROJECT_ROOT / "sources.local.json"
if _local_sources.exists():
    try:
        for _s in json.loads(_local_sources.read_text(encoding="utf-8")):
            SOURCES.append({**_s, "path": Path(_s["path"])})
    except Exception as _e:
        import sys
        print(f"config: could not load sources.local.json ({_e})", file=sys.stderr)

# Extensions to Index. .md/.pdf/.txt Get Dedicated Loaders; the Rest Load as Plain
# Text via the Fallback. An Allowlist (Not "Everything") Keeps Binaries/Images Out
INDEX_EXTENSIONS = {
    ".md", ".txt", ".pdf",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs",
    ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".sh",
}

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
# Pin the Hub Revision So a Compromised/Updated Model Repo Can't Silently Change
# What Loads (Supply-Chain) and So Re-Indexing Is Reproducible
EMBED_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
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

# Tag Filtering Is a POST-Filter (Chroma Can't Pre-Filter the Delimited tags String),
# So Over-Fetch a Bigger Candidate Pool When a Tag Is Set - Otherwise a Selective Tag
# Might Not Appear in the Small Default Top-N and Recall Would Suffer
TAG_FILTER_FETCH = 50

# Near-Duplicate Removal - Drop a Retrieved Chunk Whose Text Nearly Duplicates a
# Higher-Ranked One (Same Content Repeated Across Notes - e.g. a Project Summary in
# Both About Me and Its Own Project Note) So Top-K Spends Its Slots on DISTINCT
# Content. Token-Set Jaccard >= Floor = Duplicate. Runs on the Small Ranked Candidate
# Pool (Not the Corpus), So It Stays Cheap at Scale
DEDUP = True
DEDUP_JACCARD = 0.85

# Query Rewriting - Expand Into Alternate Phrasings (+ HyDE Answer) Before Retrieval
# So Query Wording Need Not Match the Source's
# Off by Default: Two Extra LLM Calls (~Doubles Latency); Toggle On for Weak Sources
USE_REWRITE = False
REWRITE_N = 3  # Alternate Phrasings Generated per Question

# Hybrid Retrieval - BM25 Keyword + Vector, Fused With RRF.
# DEFAULT OFF. Measured on the eval set, a Strong Dense Embedder (bge) + Cross-Encoder
# Re-Rank Beats Hybrid on Clean Prose (hit-rate 1.00/MRR 0.91 vs 0.93/0.88): BM25 Injects
# Generic Keyword Matches That the Re-Ranker Occasionally Prefers Over the Right Chunk.
# Turn ON for Keyword/Exact-Match-Heavy Corpora - Code, Acronyms, IDs, Product Names,
# OCR'd Text - Where Sparse Retrieval Catches Tokens Dense Embeddings Miss.
USE_HYBRID = False
RRF_K = 60  # RRF Smoothing Constant; Larger = Flatter Rank Weighting

# Re-Ranking - bge-reranker-base Is a Stronger Cross-Encoder Than ms-marco-MiniLM
# Reorders RETRIEVE_N Candidates Down to TOP_K
USE_RERANK = True
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"  # Pin Hub Revision (See EMBED_REVISION)
RERANK_BATCH = 64  # Scores Candidates in Batches; Matters if RETRIEVE_N Grows

# Abstention Gate - Refuse Instead of Answering When Retrieval Is Too Weak, So an
# Off-Topic or Out-of-Corpus Question Yields "Not in My Sources" Rather Than a
# Confident Hallucination From Loosely-Related Chunks. This Is the Hard Layer; the
# System Prompt's "Say You Don't Know" Is the Soft Backup.
#
# TWO SIGNALS (Answer if EITHER Is Confident, Abstain Only When Both Are Weak):
#   - Re-Ranker (Cross-Encoder): Precise on Well-Formed Queries but COLLAPSES to ~0 on
#     Terse Ones ("How Many Epochs?") - a Bare Question Doesn't Match a Table Cell.
#   - Cosine Similarity (Embedder): Topical, So It Still Separates In- vs Out-of-Corpus
#     on Those Terse Queries. Alone It Can't Tell "Related" From "Answering", Hence the OR.
# Floors Tuned on eval_set.json (Incl. Terse/Tabular Qs) - `python evaluate.py --abstain`.
# Measured Gap: Answerable Cos >= 0.576 vs Out-of-Corpus Cos <= 0.482 -> Floor at Midpoint.
USE_ABSTAIN = True
ABSTAIN_MIN_RERANK = 0.05      # Re-Rank Score That Alone Confirms Relevance (Junk Scores 0.0)
ABSTAIN_MIN_SIMILARITY = 0.53  # Cosine Floor - Catches Terse Answerable Qs Rerank Misses

# Post-Generation Grounding Check - After Generating, Ask the LLM Whether Every Claim
# in the Answer Is Supported by the Retrieved Context. Catches NEAR-DOMAIN Hallucinations
# the Retrieval Gate Structurally Can't: the Question Is On-Topic Enough to Pass the Gate,
# but the Specific Answer Isn't Actually in the Context. Costs One Extra LLM Call, So It
# Only Runs in the "Gray Zone" - Skipped When Top Rerank Is Already High (Clearly Grounded).
# NON-Destructive + Fail-Open: an "Unsupported" Verdict CAVEATS the Answer (Doesn't Delete
# It) and a Parse Failure Assumes Supported - the Local Judge Is Imperfect, So It Must Not
# Bin a Good Answer Over a False Negative or a Formatting Hiccup.
VERIFY_ANSWERS = True
VERIFY_SKIP_ABOVE = 0.9  # Skip the Check When Top Rerank >= This (Answer Is Clearly Grounded)

# LLM - Ollama, Local
OLLAMA_MODEL = "mistral"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = "30m"  # Keep Model Resident in VRAM Between Queries
