# Vector Store
import chromadb

import config
from ingest import Chunk

_client_cache = None      # Reuse One PersistentClient - Reopening the DB Is Costly
_collection_cache = None  # Reuse the Collection Handle Too


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
    global _collection_cache
    try:
        _client().delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _collection_cache = None  # Force Re-Create on Next Access


# Remove Every Chunk From One File (When It Changed or Was Removed)
def delete_source(source: str) -> None:
    _collection().delete(where={"source": source})


# Append Chunks + Vectors in Batches So Large Datasets Don't Hit the Single-Add Limit
def add(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    col = _collection()
    for i in range(0, len(chunks), config.ADD_BATCH):
        batch = chunks[i:i + config.ADD_BATCH]
        col.add(
            ids=[f"{c.source}:{c.chunk_index}" for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i:i + config.ADD_BATCH],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in batch],
        )


# Return the k Most Similar Chunks as {id, text, source, score} Dicts
def search(query_vector: list[float], k: int = config.TOP_K) -> list[dict]:
    col = _collection()
    res = col.query(query_embeddings=[query_vector], n_results=k)
    # Chroma Nests Results One Level per Query; We Only Ever Send One Query
    hits = []
    for cid, text, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "id": cid,
            "text": text,
            "source": meta["source"],
            "score": 1 - dist,  # Cosine Distance -> Similarity (Higher = Better)
        })
    return hits


# Return Every Stored Chunk as {id, text, source} - Feeds the BM25 Index
def all_chunks() -> list[dict]:
    res = _collection().get()
    return [
        {"id": cid, "text": text, "source": meta["source"]}
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
