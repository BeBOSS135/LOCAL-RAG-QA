# Vector Store
import chromadb

import config
from ingest import Chunk

_client_cache = None      # Reuse One PersistentClient - Reopening the DB Is Costly
_collection_cache = None  # Reuse the Collection Handle Too
_folders_cache = None     # Distinct folder Values, for the Scope Dropdown + Filter Build
_tags_cache = None        # Distinct tag Values, for the Tag Dropdown


def _client():
    global _client_cache
    if _client_cache is None:
        _client_cache = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client_cache


def _collection():
    global _collection_cache
    if _collection_cache is None:
        _collection_cache = _client().get_or_create_collection(
            name=config.COLLECTION_NAME,
            # Cosine Space Matches the Normalized Embeddings; HNSW Params (See Config)
            # Trade Build Time / Memory for Recall and Are Size- Not Data-Tuned
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": config.HNSW_M,
                "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                "hnsw:search_ef": config.HNSW_SEARCH_EF,
            },
        )
    return _collection_cache


# Drop the Whole Collection for a Full Rebuild
def reset() -> None:
    global _collection_cache, _folders_cache
    try:
        _client().delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _collection_cache = None  # Force Re-Create on Next Access
    _folders_cache = _tags_cache = None


# Remove Every Chunk From One File (When It Changed or Was Removed)
def delete_source(source_path: str) -> None:
    _collection().delete(where={"source_path": source_path})


# Chroma Metadata Must Be Scalar (No Lists), So tags Is Stored Delimited "|a|b|"
# - the Bars Let a Later where-substring Filter Match Whole Tags Without Prefix Bugs
# source_name (the Root) and folder (posix Dirname) Are Derived From source_path So a
# Query Can Pre-Filter to a Root/Folder via where= Before ANN - Chroma Can't Prefix-
# Match source_path Itself, and Post-Filtering After Retrieval Loses Recall at Scale
def _meta(c: Chunk) -> dict:
    sp = c.source_path
    return {
        "source_path": sp,
        "source_name": sp.split("/", 1)[0],
        "folder": sp.rsplit("/", 1)[0] if "/" in sp else sp,
        "source_type": c.source_type,
        "title": c.title,
        "heading_path": c.heading_path,
        "tags": "|" + "|".join(c.tags) + "|" if c.tags else "",
        "mtime": c.mtime,
        "chunk_index": c.chunk_index,
    }


# Build a Hit Dict Carrying the Metadata Citation + Retrieval Need Downstream
def _hit(cid: str, text: str, meta: dict, score) -> dict:
    return {
        "id": cid,
        "text": text,
        "score": score,
        "source_path": meta.get("source_path", ""),
        "source_type": meta.get("source_type", ""),
        "title": meta.get("title", ""),
        "heading_path": meta.get("heading_path", ""),
        "tags": meta.get("tags", ""),
    }


# Append Chunks + Vectors in Batches So Large Datasets Don't Hit the Single-Add Limit
def add(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    col = _collection()
    for i in range(0, len(chunks), config.ADD_BATCH):
        batch = chunks[i:i + config.ADD_BATCH]
        col.add(
            ids=[f"{c.source_path}::{c.chunk_index}" for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i:i + config.ADD_BATCH],
            metadatas=[_meta(c) for c in batch],
        )


# Return the k Most Similar Chunks as Hit Dicts (id, text, score + metadata)
# where: Optional Chroma Metadata Filter (Scope Pre-Filter) - Applied Before ANN
def search(query_vector: list[float], k: int = config.TOP_K, where: dict | None = None) -> list[dict]:
    col = _collection()
    res = col.query(query_embeddings=[query_vector], n_results=k, where=where)
    # Chroma Nests Results One Level per Query; We Only Ever Send One Query
    return [
        _hit(cid, text, meta, 1 - dist)  # Cosine Distance -> Similarity (Higher = Better)
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


# Return Every Stored Chunk (id, text + metadata, score None) - Feeds the BM25 Index
def all_chunks() -> list[dict]:
    res = _collection().get()
    return [
        _hit(cid, text, meta, None)
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]


# Distinct folder Values in the Store (e.g. "vault", "vault/Project Notes", "data").
# Powers the Scope Dropdown and the where-Filter Build. Cached - the Corpus Is Static
# Between Re-Indexes (reset() Clears It)
def list_folders() -> list[str]:
    global _folders_cache
    if _folders_cache is None:
        res = _collection().get(include=["metadatas"])
        _folders_cache = sorted({m.get("folder", "") for m in res["metadatas"] if m.get("folder")})
    return _folders_cache


# Distinct tags in the Store (Parsed From the Delimited "|a|b|" Metadata). Powers the
# Tag Dropdown. Cached; reset() Clears It. Used for a POST-Filter - Chroma Has No
# Metadata Substring/List Operator, So Tags Can't Be a where= Pre-Filter Like folders
def list_tags() -> list[str]:
    global _tags_cache
    if _tags_cache is None:
        res = _collection().get(include=["metadatas"])
        tags: set[str] = set()
        for m in res["metadatas"]:
            tags.update(t for t in m.get("tags", "").strip("|").split("|") if t)
        _tags_cache = sorted(tags)
    return _tags_cache


# Build a Chroma where= That Restricts Retrieval to a folder Prefix and Its Descendants
# (Selecting "vault" Scopes the Whole Root; "vault/Project Notes" Just That Subtree).
# Chroma Has No Prefix Operator, So Expand to an Exact $in Over the Matching folders
def scope_where(prefix: str | None) -> dict | None:
    if not prefix:
        return None
    folders = [f for f in list_folders() if f == prefix or f.startswith(prefix + "/")]
    # Impossible Sentinel on No Match -> Query Returns Nothing (Gate Then Abstains),
    # Rather Than an Empty $in Which Chroma Rejects
    return {"folder": {"$in": folders or ["\x00__no_such_folder__"]}}
