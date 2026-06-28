"""Chroma vector store wrapper: persist chunks + run similarity search."""
import chromadb

import config
from ingest import Chunk

_client_cache = None      # reuse one PersistentClient (reopening the DB is costly)
_collection_cache = None  # reuse the collection handle too


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
            # Cosine space matches the normalized embeddings; HNSW params trade
            # build time / memory for recall (see config) and are size- not data-tuned
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": config.HNSW_M,
                "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                "hnsw:search_ef": config.HNSW_SEARCH_EF,
            },
        )
    return _collection_cache


def reset() -> None:
    """Drop the whole collection (used for a full rebuild)."""
    global _collection_cache
    try:
        _client().delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _collection_cache = None  # force re-create on next access


def delete_source(source: str) -> None:
    """Remove every chunk that came from one file (used when it changed/was removed)."""
    _collection().delete(where={"source": source})


def add(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Append chunks + vectors, inserting in batches so large datasets don't blow the
    single-add limit."""
    col = _collection()
    for i in range(0, len(chunks), config.ADD_BATCH):
        batch = chunks[i:i + config.ADD_BATCH]
        col.add(
            ids=[f"{c.source}:{c.chunk_index}" for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i:i + config.ADD_BATCH],
            metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in batch],
        )


def search(query_vector: list[float], k: int = config.TOP_K) -> list[dict]:
    """Return the k most similar chunks as {id, text, source, score} dicts."""
    col = _collection()
    res = col.query(query_embeddings=[query_vector], n_results=k)
    # Chroma nests results one level per query; we only ever send one query
    hits = []
    for cid, text, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "id": cid,
            "text": text,
            "source": meta["source"],
            "score": 1 - dist,  # cosine distance -> similarity (higher = better)
        })
    return hits


def all_chunks() -> list[dict]:
    """Return every stored chunk as {id, text, source}. Feeds the BM25 index."""
    res = _collection().get()
    return [
        {"id": cid, "text": text, "source": meta["source"]}
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
